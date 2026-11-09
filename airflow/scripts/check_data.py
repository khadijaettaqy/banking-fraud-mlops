
import sys
import json
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

# Configuration HDFS
HDFS_URL = "http://namenode:9870"

DATA_PATH = "/fraud/raw/transactions_v2"

url = (
    f"{HDFS_URL}/webhdfs/v1{DATA_PATH}"
    "?op=LISTSTATUS"
)

try:
    # Lire le contenu du dossier HDFS
    with urlopen(url, timeout=15) as response:
        data = json.load(response)

    files = data["FileStatuses"]["FileStatus"]

    # Sélectionner uniquement les fichiers Parquet
    parquet_files = [
        file["pathSuffix"]
        for file in files
        if file["pathSuffix"].endswith(".parquet")
    ]

    if not parquet_files:
        print("Aucun fichier Parquet disponible dans HDFS.")
        sys.exit(99)

    print("Des données sont disponibles dans HDFS.")
    print("Nombre de fichiers détectés :", len(parquet_files))

except HTTPError as e:
    print(f"Erreur HTTP HDFS : {e}")
    sys.exit(1)

except URLError as e:
    print(f"Connexion HDFS impossible : {e}")
    sys.exit(1)
