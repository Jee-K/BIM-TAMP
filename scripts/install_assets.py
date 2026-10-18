#!/usr/bin/env python3
"""Install an authorized, exact copy of the experiment's robot mesh bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--from', dest='source', required=True, type=Path)
a = p.parse_args()
assets = json.loads((ROOT/'models/ASSET_MANIFEST.json').read_text())['assets']
for item in assets:
    src = a.source/item['path']
    if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest() != item['sha256']:
        p.error('Missing or mismatched mesh: '+str(src))
for item in assets:
    src = (a.source/item['path']).resolve()
    dst = ROOT/'models'/item['path']
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src != dst.resolve():
        shutil.copy2(src, dst)
print('Verified and installed', len(assets), 'mesh files.')
