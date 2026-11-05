#!/bin/bash
# Build a self-contained scale-out kit from a committed revision (default HEAD):
#   DEST/            the files of examples/slurm_scripts/scale_out at REVISION
#   DEST/aweSOM-src/ git archive REVISION src/aweSOM, unmodified
#   DEST/kit-manifest.json  commit, branch, SHA256 of every package .py file and kit file
# Usage: make_kit.sh DEST [REVISION]. DEST must not exist.
set -eo pipefail
dest=$1; rev=${2:-HEAD}
[ -n "$dest" ] || { echo "usage: make_kit.sh DEST [REVISION]" >&2; exit 2; }
[ ! -e "$dest" ] || { echo "DEST exists: $dest" >&2; exit 2; }
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
commit=$(git -C "$repo" rev-parse "$rev^{commit}")
mkdir -p "$dest/aweSOM-src"
git -C "$repo" archive "$commit" examples/slurm_scripts/scale_out | tar -x -C "$dest" --strip-components=3
git -C "$repo" archive "$commit" src/aweSOM | tar -x -C "$dest/aweSOM-src"
rm -f "$dest/make_kit.sh"
python3 - "$repo" "$commit" "$dest" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
repo, commit, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sha = lambda b: hashlib.sha256(b).hexdigest()
pkg = dest / "aweSOM-src" / "src" / "aweSOM"
files = {str(p.relative_to(pkg)): sha(p.read_bytes()) for p in sorted(pkg.rglob("*.py"))}
kit = {p.name: sha(p.read_bytes()) for p in sorted(dest.iterdir()) if p.is_file()}
branches = subprocess.check_output(["git", "-C", repo, "branch", "--contains", commit,
                                    "--format=%(refname:short)"], text=True).split()
m = dict(commit=commit, branch=",".join(branches), files=files, kit_files=kit,
         tree_sha256=sha(json.dumps(files, sort_keys=True).encode()))
(dest / "kit-manifest.json").write_text(json.dumps(m, indent=1) + "\n")
print(f"kit at {dest}: commit {commit[:12]} ({m['branch']}), {len(files)} package files, {len(kit)} kit files")
PY
