"""
Générateur de transactions bancaires avec fraude, pour l'entraînement de modèles ML.

Différences clés par rapport à la v1 :
  * Profils clients persistants (montant moyen, pays d'origine) -> features relatives
    (amount_ratio, is_foreign par rapport au pays du client).
  * tx_last_hour et time_since_last_tx_min sont CALCULÉS sur la vraie chronologie
    du client (au lieu d'être tirés au hasard).
  * Plusieurs scénarios de fraude réalistes (card testing, prise de contrôle de
    compte, carte volée à l'étranger, fraude "discrète" difficile à voir).
  * Chevauchement réaliste : des transactions légitimes "suspectes" (voyage,
    gros achat, nuit) -> pas de bruit artificiel sur les labels.
  * Bruit de labels très faible (0.5 %) au lieu de 3 % symétrique (qui faisait
    ~22 % de faux positifs parmi les labels "fraude").

Usage :
    python generate_fraud_data.py --output csv --evaluate
    python generate_fraud_data.py --output kafka
    python generate_fraud_data.py --output both --n 50000
"""
import argparse
import csv
import json
import math
import random
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone

COUNTRIES = ["MA", "FR", "ES", "US"]
TYPES = ["TRANSFER", "PAYMENT", "WITHDRAWAL"]


# ============================
# OUTILS
# ============================
def day_hour(night_bias):
    """Heure tirée avec un biais jour (normal) ou nuit (fraude)."""
    if night_bias:
        w = [6 if (h < 6 or h >= 22) else 1 for h in range(24)]
    else:
        w = [1 if (h < 6 or h >= 22) else 6 for h in range(24)]
    return random.choices(range(24), weights=w)[0]


def random_ts(now, span_days, night_bias):
    base = now - timedelta(days=random.uniform(0.05, span_days))
    return base.replace(
        hour=day_hour(night_bias),
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
        microsecond=0,
    )


def foreign_country(home):
    return random.choice([c for c in COUNTRIES if c != home])


def make_profiles(n_customers):
    profiles = {}
    for i in range(1, n_customers + 1):
        profiles[f"CUST{i:04d}"] = {
            "home": random.choices(COUNTRIES, weights=[90, 5, 3, 2])[0],
            "mean_amount": random.lognormvariate(6.0, 0.5),  # ~400 MAD
        }
    return profiles


def tx(cid, ts, amount, ttype, country, new_device, new_benef, fraud):
    return {
        "customer_id": cid,
        "ts": ts,
        "amount": round(min(max(amount, 1.0), 50000), 2),
        "transaction_type": ttype,
        "country": country,
        "new_device": int(new_device),
        "new_beneficiary": int(new_benef and ttype == "TRANSFER"),
        "is_fraud": fraud,
    }


# ============================
# TRANSACTIONS NORMALES
# ============================
def normal_tx(cid, p, now, span):
    ts = random_ts(now, span, night_bias=False)
    if random.random() < 0.05:                      # 5 % d'activité nocturne légitime
        ts = ts.replace(hour=random.choice([22, 23, 0, 1, 2, 3, 4, 5]))
    amount = random.lognormvariate(math.log(p["mean_amount"]), 0.7)
    if random.random() < 0.03:                      # gros achat légitime
        amount *= random.uniform(3, 6)
    ttype = random.choices(TYPES, weights=[20, 55, 25])[0]
    country = p["home"] if random.random() < 0.95 else foreign_country(p["home"])  # voyage
    return tx(cid, ts, amount, ttype, country,
              new_device=random.random() < 0.03,
              new_benef=random.random() < 0.15,
              fraud=0)


# ============================
# SCÉNARIOS DE FRAUDE
# ============================
def fraud_episode(cid, p, now, span):
    pattern = random.choices(
        ["card_testing", "takeover", "stolen_abroad", "stealth"],
        weights=[25, 35, 25, 15],
    )[0]
    ts = random_ts(now, span, night_bias=True)
    home, mean = p["home"], p["mean_amount"]
    out = []

    if pattern == "card_testing":        # rafale de petits montants
        country = foreign_country(home) if random.random() < 0.6 else home
        for _ in range(random.randint(3, 6)):
            ts += timedelta(seconds=random.randint(10, 120))
            out.append(tx(cid, ts, random.uniform(1, 40), "PAYMENT", country,
                          random.random() < 0.5, False, 1))

    elif pattern == "takeover":          # compte piraté : gros virement, nouvel appareil
        for _ in range(random.randint(1, 3)):
            ts += timedelta(minutes=random.randint(1, 15))
            ttype = random.choices(["TRANSFER", "WITHDRAWAL"], weights=[60, 40])[0]
            country = foreign_country(home) if random.random() < 0.5 else home
            out.append(tx(cid, ts, mean * random.uniform(3, 12), ttype, country,
                          random.random() < 0.9, random.random() < 0.9, 1))

    elif pattern == "stolen_abroad":     # carte volée utilisée à l'étranger
        country = foreign_country(home)
        for _ in range(random.randint(1, 3)):
            ts += timedelta(minutes=random.randint(5, 60))
            out.append(tx(cid, ts, mean * random.uniform(1.5, 5),
                          random.choice(["PAYMENT", "WITHDRAWAL"]), country,
                          random.random() < 0.2, False, 1))

    else:                                # fraude discrète : ressemble à du normal
        ts = ts.replace(hour=day_hour(False))
        out.append(tx(cid, ts, mean * random.lognormvariate(0, 0.5),
                      random.choices(TYPES, weights=[40, 40, 20])[0], home,
                      random.random() < 0.3, random.random() < 0.4, 1))
    return out


# ============================
# FEATURES D'HISTORIQUE
# ============================
def add_history_features(rows, profiles):
    by_cust = defaultdict(list)
    for r in rows:
        by_cust[r["customer_id"]].append(r)

    for cid, lst in by_cust.items():
        lst.sort(key=lambda r: r["ts"])
        start = 0
        for i, r in enumerate(lst):
            while lst[start]["ts"] < r["ts"] - timedelta(hours=1):
                start += 1
            r["tx_last_hour"] = i - start          # nb de tx sur la dernière heure (hors courante)
            if i == 0:
                r["time_since_last_tx_min"] = 10080.0
            else:
                gap = (r["ts"] - lst[i - 1]["ts"]).total_seconds() / 60
                r["time_since_last_tx_min"] = round(min(gap, 10080.0), 2)

    for r in rows:
        p = profiles[r["customer_id"]]
        r["amount_ratio"] = round(r["amount"] / p["mean_amount"], 3)
        r["is_foreign"] = int(r["country"] != p["home"])
        r["hour"] = r["ts"].hour
        r["is_night"] = int(r["hour"] < 6 or r["hour"] >= 22)


# ============================
# GÉNÉRATION GLOBALE
# ============================
def generate(n, fraud_rate, span, n_customers, label_noise):
    now = datetime.now(timezone.utc)
    profiles = make_profiles(n_customers)
    cids = list(profiles)

    n_fraud = round(n * fraud_rate)
    n_normal = n - n_fraud

    rows = []
    for _ in range(n_normal):
        cid = random.choice(cids)
        rows.append(normal_tx(cid, profiles[cid], now, span))

    frauds = []
    while len(frauds) < n_fraud:
        cid = random.choice(cids)
        frauds.extend(fraud_episode(cid, profiles[cid], now, span))
    rows.extend(frauds[:n_fraud])

    add_history_features(rows, profiles)

    for r in rows:
        r["transaction_id"] = str(uuid.uuid4())
        r["timestamp"] = r.pop("ts").isoformat()
        if random.random() < label_noise:             # très léger bruit de labels
            r["is_fraud"] = 1 - r["is_fraud"]

    rows.sort(key=lambda r: r["timestamp"])
    return rows


FIELDS = [
    "transaction_id", "customer_id", "timestamp", "amount", "amount_ratio",
    "transaction_type", "country", "is_foreign", "hour", "is_night",
    "tx_last_hour", "time_since_last_tx_min", "new_device", "new_beneficiary",
    "is_fraud",
]


# ============================
# SORTIES
# ============================
def to_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})
    print(f"CSV écrit : {path}")


def to_kafka(rows, topic, servers):
    from kafka import KafkaProducer  # import tardif : pas requis pour le mode CSV
    producer = KafkaProducer(
        bootstrap_servers=servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        retries=5,
    )
    try:
        for i, r in enumerate(rows, 1):
            producer.send(topic, key=r["customer_id"], value={k: r[k] for k in FIELDS})
            if i % 500 == 0:
                print(f"{i}/{len(rows)} transactions envoyées")
    finally:
        producer.flush()
        producer.close()


# ============================
# ÉVALUATION RAPIDE
# ============================
def evaluate(rows):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.DataFrame(rows)[FIELDS]
    X = pd.get_dummies(
        df.drop(columns=["transaction_id", "customer_id", "timestamp", "is_fraud"]),
        columns=["transaction_type", "country"],
    )
    y = df["is_fraud"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0
    ).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    print("\n=== ÉVALUATION (RandomForest) ===")
    print("ROC-AUC :", round(roc_auc_score(yte, proba), 4))
    print("PR-AUC  :", round(average_precision_score(yte, proba), 4))
    print(classification_report(yte, (proba >= 0.5).astype(int), digits=3))


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--fraud-rate", type=float, default=0.10)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--customers", type=int, default=1000)
    ap.add_argument("--label-noise", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", choices=["csv", "kafka", "both"], default="csv")
    ap.add_argument("--csv-path", default="transactions.csv")
    ap.add_argument("--topic", default="banking-transactions-v2")
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed)
    rows = generate(a.n, a.fraud_rate, a.days, a.customers, a.label_noise)

    n_fraud = sum(r["is_fraud"] for r in rows)
    print(f"{len(rows)} transactions | fraudes : {n_fraud} ({100 * n_fraud / len(rows):.2f} %)")

    if a.output in ("csv", "both"):
        to_csv(rows, a.csv_path)
    if a.output in ("kafka", "both"):
        to_kafka(rows, a.topic, [a.bootstrap])
    if a.evaluate:
        evaluate(rows)