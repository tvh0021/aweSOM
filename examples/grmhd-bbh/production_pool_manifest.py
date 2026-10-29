"""Write deployment hashes from an explicit committed aweSOM revision."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def generate(root, revision):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    commit = git('rev-parse', revision+'^{commit}').decode().strip()
    paths = git('ls-tree', '-r', '--name-only', commit, 'src/aweSOM/').decode().splitlines()
    return dict(commit=commit,
                files={str(Path(p).relative_to('src/aweSOM')):
                       hashlib.sha256(git('show', f'{commit}:{p}')).hexdigest() for p in paths},
                driver_sha256=hashlib.sha256(git('show', f'{commit}:examples/grmhd-bbh/production_pool.py')).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('revision')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(args.root,args.revision), indent=2)+'\n')
