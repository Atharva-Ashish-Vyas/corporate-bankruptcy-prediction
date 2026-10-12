"""Recover the official data file with checksum verification."""
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
URL="https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
EXPECTED="662617d2f90bfd17223595e50a0ac7007f9ee7915cfb519dee0422b31dd13ebc"
def main():
    with urllib.request.urlopen(URL,timeout=60) as response:
        content=response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        raw=archive.read("3year.arff")
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:
        raise ValueError("Official file checksum changed. Inspect the new data before using it.")
    target=ROOT/"data/raw/3year.arff"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(raw)
    print(f"Verified and saved {target}")
if __name__=="__main__":main()
