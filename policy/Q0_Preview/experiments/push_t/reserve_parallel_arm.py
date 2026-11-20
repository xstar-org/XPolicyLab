"""Prevent an already-running serial driver from enqueueing a delegated arm.

Run only after explicit operator authorization to parallelize a second arm.
The immutable-launcher check in cluster_control rejects the reserved serial
submission before it can create a queue request. No active service is stopped.
"""
import hashlib
import json
from pathlib import Path
import re
import sys


def reserve(message):
    root = Path(message['root']).resolve()
    serial, parallel = message['serial_service'], message['parallel_service']
    if message.get('operator_authorized_parallel') is not True:
        raise ValueError('Explicit parallel operator authorization is required')
    for name in (serial, parallel):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}-official_only', name):
            raise ValueError('Reservation is restricted to official_only services')
    if serial == parallel:
        raise ValueError('Delegated service must have a separate identity')
    if any((root/folder/serial).exists() for folder in ('runs',)) or any(
            (root/folder/(serial+'.json')).exists() for folder in ('requests','claims','results')):
        raise ValueError('Cannot reserve an already submitted or active serial arm')
    content = ('#!/bin/bash\n# Reserved by explicit parallel evaluation instruction.\n'
               f'# Delegate: {parallel}\n'
               'echo "This serial arm was delegated; use its separate parallel archive." >&2\nexit 73\n').encode()
    launcher = root/'code'/('launch-'+serial+'.sh')
    if launcher.exists():
        if launcher.read_bytes() != content:
            raise ValueError('Existing launcher is not this delegation reservation')
    else:
        with launcher.open('xb') as file:
            file.write(content)
        launcher.chmod(0o600)
    record = dict(serial_service=serial, parallel_service=parallel,
        launcher=str(launcher), launcher_sha256=hashlib.sha256(content).hexdigest(),
        prevents_enqueue_via='cluster_control immutable launcher conflict check',
        operator_authorized_parallel=True, active_original_untouched=True)
    certificate = root/'code'/('delegation-'+serial+'.json')
    encoded = json.dumps(record, indent=2).encode()
    if certificate.exists():
        if certificate.read_bytes() != encoded:
            raise ValueError('Conflicting delegation certificate')
    else:
        with certificate.open('xb') as file:
            file.write(encoded)
    return record


if __name__ == '__main__':
    print(json.dumps(reserve(json.load(sys.stdin))))
