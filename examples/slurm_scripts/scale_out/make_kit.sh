#!/bin/bash
# Build a self-contained scale-out kit from a committed revision (default HEAD):
#   DEST/            the files of examples/slurm_scripts/scale_out at REVISION
#   DEST/aweSOM-src/ git archive REVISION src/aweSOM, unmodified
#   DEST/wheels/     copy of WHEELS: the binary mpi4py wheel (MPI_BUILD=wheel, the default)
#                    and optionally the sdist and its build wheels (MPI_BUILD=source); compute
#                    nodes have no internet, so probe.sbatch installs mpi4py from these (README, "MPI")
#   DEST/kit-manifest.json  commit, branch, SHA256 of every package .py file, kit file and wheel
# Usage: [WHEELS=DIR] make_kit.sh DEST [REVISION]. DEST must not exist. WHEELS (default below)
# must hold SHA256SUMS and the cp311 x86_64 manylinux mpi4py wheel; if it holds the sdist
# mpi4py-*.tar.gz it must also hold a cython wheel. Every file is checked against SHA256SUMS.
set -eo pipefail
dest=$1; rev=${2:-HEAD}
[ -n "$dest" ] || { echo "usage: make_kit.sh DEST [REVISION]" >&2; exit 2; }
[ ! -e "$dest" ] || { echo "DEST exists: $dest" >&2; exit 2; }
wheels=${WHEELS:-/Volumes/ExternalNVMe/awesom-results/planning/2026-11-05/scale-out/mpi/wheels}
[ -f "$wheels/SHA256SUMS" ] || { echo "WHEELS has no SHA256SUMS: $wheels" >&2; exit 2; }
ls "$wheels"/mpi4py-*-cp311-cp311-*manylinux*_x86_64*.whl >/dev/null 2>&1 \
  || { echo "WHEELS has no binary mpi4py-*-cp311-*manylinux*_x86_64.whl: $wheels" >&2; exit 2; }
if ls "$wheels"/mpi4py-*.tar.gz >/dev/null 2>&1; then
  ls "$wheels"/[cC]ython-*.whl >/dev/null 2>&1 || { echo "WHEELS has the mpi4py sdist but no cython wheel: $wheels" >&2; exit 2; }
fi
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
commit=$(git -C "$repo" rev-parse "$rev^{commit}")
mkdir -p "$dest/aweSOM-src"
git -C "$repo" archive "$commit" examples/slurm_scripts/scale_out | tar -x -C "$dest" --strip-components=3
git -C "$repo" archive "$commit" src/aweSOM | tar -x -C "$dest/aweSOM-src"
rm -f "$dest/make_kit.sh"
mkdir "$dest/wheels"
cp -p "$wheels"/* "$dest/wheels/"
python3 - "$repo" "$commit" "$dest" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
repo, commit, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sha = lambda b: hashlib.sha256(b).hexdigest()
pkg = dest / "aweSOM-src" / "src" / "aweSOM"
files = {str(p.relative_to(pkg)): sha(p.read_bytes()) for p in sorted(pkg.rglob("*.py"))}
kit = {p.name: sha(p.read_bytes()) for p in sorted(dest.iterdir()) if p.is_file()}
wdir = dest / "wheels"
wheels = {p.name: sha(p.read_bytes()) for p in sorted(wdir.iterdir()) if p.is_file() and p.name != "SHA256SUMS"}
sums = dict(reversed(line.split()) for line in (wdir / "SHA256SUMS").read_text().splitlines() if line.strip())
if sums != wheels:
    sys.exit(f"wheels do not match SHA256SUMS: files {sorted(wheels)} vs listed {sorted(sums)}, "
             f"bad {sorted(k for k in wheels if sums.get(k) != wheels[k])}")
branches = subprocess.check_output(["git", "-C", repo, "branch", "--contains", commit,
                                    "--format=%(refname:short)"], text=True).split()
m = dict(commit=commit, branch=",".join(branches), files=files, kit_files=kit, wheels=wheels,
         tree_sha256=sha(json.dumps(files, sort_keys=True).encode()))
(dest / "kit-manifest.json").write_text(json.dumps(m, indent=1) + "\n")
print(f"kit at {dest}: commit {commit[:12]} ({m['branch']}), {len(files)} package files, {len(kit)} kit files, {len(wheels)} wheels")
PY
