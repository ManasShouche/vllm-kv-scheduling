"""Recompute research diagnostics from the three checksummed matched archives.

Run from the project directory with Python >=3.12 and requirements-client.txt.
Uses a temporary extraction; never edits the archived historical experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from lab.research import batch_analysis, plot_surfaces, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New output directory')
    args = parser.parse_args()
    batches = []
    with tempfile.TemporaryDirectory(prefix='kv-research-') as temporary:
        for run in ('matched-20261007a', 'improvement-20261007a', 'replication-20261007a'):
            source = Path('evidence')/run
            archive = source/'results.tar.gz'
            expected = (source/'results.tar.gz.sha256').read_text().split()[0]
            if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Archive checksum mismatch: {run}')
            destination = Path(temporary)/run
            with tarfile.open(archive) as stream:
                stream.extractall(destination, filter='data')
            root = destination/'kv-pressure-run/kv-pressure-lab'
            batch = batch_analysis(root/'results'/f'{run}-matched', root/'workloads'/run)
            batches.append(batch)
    write_report(batches, args.output)
    plot_surfaces(batches, args.output)
    print(json.dumps({'output': args.output, 'matched_batches': len(batches),
                      'new_gpu_execution': False}))


if __name__ == '__main__':
    main()
