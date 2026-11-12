#!/usr/bin/env bash
set -e
cd ~/dgg2_codegraph
PYTHON=~/dg-venv/bin/python

echo "=== [SPARK2] Starting Training nondg_frozen_inst_s42 (4,824 steps) ==="
$PYTHON src/cg_train_nondg_expB.py --arm nondg_frozen_inst_s42 --epochs 3 --batch-size 8 --device cuda --eval-every 200 2>&1 | tee train_frozen_inst_s42_expB.log

echo "=== [SPARK2] Evaluating nondg_frozen_inst_s42 on all manifests ==="
$PYTHON src/cg_eval_nondg_expB.py --arm nondg_frozen_inst_s42 --device cuda 2>&1 | tee eval_frozen_inst_s42_expB.log

echo "=== [SPARK2] Evaluating nondg_enc_lora_s42 on all manifests ==="
$PYTHON src/cg_eval_nondg_expB.py --arm nondg_enc_lora_s42 --device cuda 2>&1 | tee eval_enc_lora_s42_expB.log

echo "=== [SPARK2] All Exp B jobs completed successfully! ==="
