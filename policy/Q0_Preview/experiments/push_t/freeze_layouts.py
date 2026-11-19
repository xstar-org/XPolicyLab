"""Freeze and verify the exact 50 official layout files without launching the simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def layout_manifest(directory):
    pattern = re.compile(r'push_T_(\d+)\.json')
    paths = sorted((p for p in directory.iterdir() if pattern.fullmatch(p.name)),
                   key=lambda p: int(pattern.fullmatch(p.name).group(1)))
    if len(paths) != 50:
        raise ValueError(f'Expected exactly 50 standard push_T layouts, found {len(paths)}')
    return dict(task='push_T', seed=1, layout_count=50, layouts=[dict(
        layout_id=i, filename=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for i, p in enumerate(paths)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layout-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    actual = layout_manifest(args.layout_dir)
    if args.verify:
        if json.loads(args.manifest.read_text()) != actual:
            raise ValueError('Layouts changed between experiment arms')
    else:
        with args.manifest.open('x') as file:
            json.dump(actual, file, indent=2)


if __name__ == '__main__':
    main()
