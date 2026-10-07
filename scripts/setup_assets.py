#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_assets.py — Telecharge les dependances front non committees.

Plotly.js (~3.6 Mo) n'est pas inclus dans le depot (librairie externe).
Ce script le telecharge dans static/.

Usage :
    python scripts/setup_assets.py
"""

import os
import sys
import urllib.request
from pathlib import Path

PLOTLY_VERSION = "2.32.0"
PLOTLY_URL = f"https://cdn.plot.ly/plotly-{PLOTLY_VERSION}.min.js"

ROOT = Path(__file__).parent.parent.resolve()
STATIC = ROOT / "static"
TARGET = STATIC / "plotly.min.js"


def main():
    STATIC.mkdir(exist_ok=True)
    if TARGET.exists():
        print(f"Plotly deja present : {TARGET}")
        return
    print(f"Telechargement de Plotly.js v{PLOTLY_VERSION}...")
    try:
        urllib.request.urlretrieve(PLOTLY_URL, TARGET)
        size_mb = TARGET.stat().st_size / 1e6
        print(f"OK : {TARGET} ({size_mb:.1f} Mo)")
    except Exception as e:
        print(f"ERREUR : {e}")
        print(f"Telecharge manuellement {PLOTLY_URL}")
        print(f"et place-le dans {TARGET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
