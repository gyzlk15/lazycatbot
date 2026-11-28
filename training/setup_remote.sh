#!/usr/bin/env bash
# LazyCatBot remote training setup (Debian 13, no system-wide pip).
#
# Usage:
#   ./training/setup_remote.sh                 # repo root inferred from this file
#   ./training/setup_remote.sh /path/to/repo   # explicit project root
#
# Creates .venv in the project root, installs training/requirements-remote.txt
# into it, then checks nvidia-smi and a real CUDA + bf16 tensor operation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
VENV="${PROJECT_ROOT}/.venv"
REQ="${PROJECT_ROOT}/training/requirements-remote.txt"

cd "${PROJECT_ROOT}"

echo "== LazyCatBot remote setup =="
echo "project: ${PROJECT_ROOT}"
echo "python:  ${PYTHON}"

if [ ! -f "${REQ}" ]; then
    echo "ERROR: ${REQ} not found (run from the repo or pass the project root)." >&2
    exit 1
fi
if [ ! -f "${PROJECT_ROOT}/training/train_halfkp.py" ]; then
    echo "ERROR: training/train_halfkp.py missing; bundle is incomplete." >&2
    exit 1
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON} not found." >&2
    echo "  Debian: sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip" >&2
    exit 1
fi

"${PYTHON}" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python >= 3.11 required, found {sys.version.split()[0]}")
print("python", sys.version.split()[0])
PY

if [ ! -d "${VENV}" ]; then
    echo "creating venv at ${VENV}"
    if ! "${PYTHON}" -m venv "${VENV}"; then
        echo "ERROR: venv creation failed." >&2
        echo "  Debian: sudo apt-get install -y python3-venv" >&2
        exit 1
    fi
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ}"

echo
echo "== disk =="
df -h "${PROJECT_ROOT}" || true
echo
echo "== CPU / RAM =="
nproc || true
free -h || true
echo
echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found; CPU training still works but is slow." >&2
fi

echo
echo "== torch / CUDA smoke test =="
python - <<'PY'
import sys
import torch

print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available; --device cuda will fail.", file=sys.stderr)
    sys.exit(0)

props = torch.cuda.get_device_properties(0)
print("device:", torch.cuda.get_device_name(0))
print(f"vram: {props.total_memory / 2**30:.1f} GiB, sm_{props.major}{props.minor}")
print("bf16 supported:", torch.cuda.is_bf16_supported())

a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
print("bf16 matmul ok:", tuple(c.shape), float(c.float().abs().mean()))

emb = torch.nn.EmbeddingBag(49152, 256, mode="sum", sparse=True).cuda()
idx = torch.randint(0, 49152, (4096,), device="cuda")
off = torch.arange(0, 4064, 32, device="cuda")
out = emb(idx, off)
loss = out.float().pow(2).mean()
loss.backward()
opt = torch.optim.SparseAdam(emb.parameters(), lr=1e-3)
opt.step()
print("sparse EmbeddingBag + SparseAdam ok")
PY

cat <<EOF

Setup complete.

Next steps (see training/RUNBOOK_A6000.md):
  source ${VENV}/bin/activate
  python training/train_halfkp.py --smoke --out /tmp/lcb_smoke
  python training/train_halfkp.py --profile teacher --data /data/pilot_shards \\
      --out /data/teacher_model --data-on-gpu --amp bf16

EOF
