#!/usr/bin/env bash
# SymFold 环境设置脚本
# 用法: bash scripts/setup.sh
set -e

echo "=== SymFold Setup ==="

# 1. 创建必要目录
echo "[1/4] Creating directories..."
mkdir -p ckpt/cond_ckpt data model logs output

# 2. 检查 Python 环境
echo "[2/4] Checking Python environment..."
python --version || { echo "ERROR: python not found"; exit 1; }
python -c "import torch; print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not installed. Run:"
    echo "  pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
    exit 1
}
python -c "import einops" || { echo "Installing einops..."; pip install einops; }

# 3. 检查预训练权重
echo "[3/4] Checking pretrained weights..."
if [ ! -f ckpt/cond_ckpt/RNA-FM_pretrained.pth ]; then
    echo "WARNING: ckpt/cond_ckpt/RNA-FM_pretrained.pth not found!"
    echo "  Please download RNA-FM weights from: https://github.com/ml4bio/RNA-FM"
    echo "  And place it at: ckpt/cond_ckpt/RNA-FM_pretrained.pth"
fi
if [ ! -f ckpt/cond_ckpt/ufold_train_alldata.pt ]; then
    echo "WARNING: ckpt/cond_ckpt/ufold_train_alldata.pt not found!"
    echo "  Please download UFold weights from: https://github.com/uci-cbcl/UFold"
    echo "  And place it at: ckpt/cond_ckpt/ufold_train_alldata.pt"
fi

# 4. 检查数据
echo "[4/4] Checking data..."
if [ ! -d data/preprocess ]; then
    echo "WARNING: data/preprocess/ not found!"
    echo "  Please prepare training data. See README.md for data sources."
    echo "  Expected structure:"
    echo "    data/preprocess/RNAStrAlign/*.cPickle"
    echo "    data/preprocess/bpRNA/*.cPickle"
    echo "    data/preprocess/bpRNA-new/*.cPickle"
    echo "    data/bpRNA/VL0.cPickle (validation)"
    echo "    data/bpRNA/TS0.cPickle (test)"
    echo "    data/RNAStrAlign/test.cPickle"
    echo "    data/ArchiveII/archiveII.cPickle"
    echo "    data/PDB/TS1.cPickle, TS2.cPickle, TS3.cPickle, TS_hard.cPickle"
fi

# 5. Quick import test
echo ""
echo "=== Import Test ==="
python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from src.data import SimpleRNADataset, build_index
from src.gpu_features import get_data_fcn_gpu
from src.physics_energy import PhysicsGuidance
print('All imports OK!')
" && echo "Setup complete!" || echo "Import test failed. Check error above."
