#!/usr/bin/env python3
"""Download the small GeoNames datasets used for offline image enrichment."""

from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import io
import ssl

try:
    import certifi
except ImportError:
    certifi = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "geonames"
BASE = "https://download.geonames.org/export/dump"


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "witchcraft-offline-geotagger/1"})
    candidates = [
        certifi.where() if certifi else None,
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ]
    cafile = next((path for path in candidates if path and Path(path).is_file()), None)
    context = ssl.create_default_context(cafile=cafile)
    with urlopen(request, context=context) as response:
        return response.read()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cities = OUTPUT / "cities5000.txt"
    countries = OUTPUT / "countryInfo.txt"
    regions = OUTPUT / "admin1CodesASCII.txt"
    if not cities.exists():
        archive = ZipFile(io.BytesIO(download(f"{BASE}/cities5000.zip")))
        cities.write_bytes(archive.read("cities5000.txt"))
    if not countries.exists():
        countries.write_bytes(download(f"{BASE}/countryInfo.txt"))
    if not regions.exists():
        regions.write_bytes(download(f"{BASE}/admin1CodesASCII.txt"))
    (OUTPUT / "README.txt").write_text(
        "GeoNames data is licensed under CC BY 4.0.\n"
        "Source: https://www.geonames.org/\n",
        encoding="utf-8",
    )
    print(f"offline geotag assets ready in {OUTPUT}")


if __name__ == "__main__":
    main()
