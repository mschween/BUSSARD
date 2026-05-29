import csv
import gzip
import json
import sqlite3
import subprocess
from pathlib import Path

CSV_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
CSV_PATH = Path("data/conceptnet-assertions-5.7.0.csv.gz")
DB_PATH = Path("data/conceptnet_normalized.db")

BASE = "http://conceptnet.io"
RELATIONS = {"/r/RelatedTo", "/r/PartOf"}
LANGUAGE = "/c/en/"


def to_full_rel_url(uri):
    # CSV uses short form (/r/RelatedTo); mapper queries full form
    return BASE + uri


def to_full_node_url(uri):
    # CSV node may carry a POS/sense suffix (/c/en/london/n); mapper queries /c/en/london.
    # Keep only term: drop everything after /c/en/<term>.
    parts = uri.split("/")  # ['', 'c', 'en', 'london', 'n']
    return BASE + "/".join(parts[:4])


def download_csv():
    if CSV_PATH.exists():
        print(f"CSV already exists at {CSV_PATH}, skipping download.")
        return
    print("Downloading ConceptNet CSV...")
    subprocess.run(["wget", "-c", "-O", str(CSV_PATH), CSV_URL], check=True)
    print("Download complete.")


def build_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS node_norm (
            node_pk INTEGER PRIMARY KEY AUTOINCREMENT,
            node_url TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS rel_norm (
            rel_pk INTEGER PRIMARY KEY AUTOINCREMENT,
            rel_url TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS edge_norm (
            edge_pk INTEGER PRIMARY KEY AUTOINCREMENT,
            start_fk INTEGER NOT NULL REFERENCES node_norm(node_pk),
            end_fk   INTEGER NOT NULL REFERENCES node_norm(node_pk),
            rel_fk   INTEGER NOT NULL REFERENCES rel_norm(rel_pk),
            weight   REAL NOT NULL
        );
    """
    )

    for rel in RELATIONS:
        cursor.execute(
            "INSERT OR IGNORE INTO rel_norm (rel_url) VALUES (?)",
            (to_full_rel_url(rel),),
        )
    conn.commit()

    node_cache = {}
    rel_cache = {}

    def get_node_pk(url):
        if url not in node_cache:
            cursor.execute(
                "INSERT OR IGNORE INTO node_norm (node_url) VALUES (?)", (url,)
            )
            cursor.execute("SELECT node_pk FROM node_norm WHERE node_url = ?", (url,))
            node_cache[url] = cursor.fetchone()[0]
        return node_cache[url]

    def get_rel_pk(url):
        if url not in rel_cache:
            cursor.execute("SELECT rel_pk FROM rel_norm WHERE rel_url = ?", (url,))
            rel_cache[url] = cursor.fetchone()[0]
        return rel_cache[url]

    print("Building DB from CSV (English RelatedTo/PartOf edges only)...")
    count = 0
    with gzip.open(CSV_PATH, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            _, rel_url, start_url, end_url, json_info = row
            if rel_url not in RELATIONS:
                continue
            if not start_url.startswith(LANGUAGE) or not end_url.startswith(LANGUAGE):
                continue

            weight = json.loads(json_info).get("weight", 1.0)
            start_pk = get_node_pk(to_full_node_url(start_url))
            end_pk = get_node_pk(to_full_node_url(end_url))
            rel_pk = get_rel_pk(to_full_rel_url(rel_url))

            cursor.execute(
                "INSERT INTO edge_norm (start_fk, end_fk, rel_fk, weight) VALUES (?, ?, ?, ?)",
                (start_pk, end_pk, rel_pk, weight),
            )
            count += 1
            if count % 100_000 == 0:
                conn.commit()
                print(f"  {count} edges inserted...")

    conn.commit()

    print("Creating indexes...")
    cursor.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_edge_start ON edge_norm(start_fk);
        CREATE INDEX IF NOT EXISTS idx_edge_rel   ON edge_norm(rel_fk);
    """
    )
    conn.commit()
    conn.close()
    print(f"Done. DB saved to {DB_PATH} with {count} edges.")


if __name__ == "__main__":
    download_csv()
    build_db()
