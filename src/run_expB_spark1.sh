#!/usr/bin/env bash
set -e
cd ~/dgg2_codegraph
PYTHON=~/dg-venv/bin/python

echo "=== [SPARK1] Starting Training nondg_enc_lora_s1337 (4,824 steps) ==="
$PYTHON src/cg_train_nondg_expB.py --arm nondg_enc_lora_s1337 --epochs 3 --batch-size 8 --device cuda --eval-every 200 2>&1 | tee train_enc_lora_s1337_expB.log

echo "=== [SPARK1] Evaluating nondg_enc_lora_s1337 on all manifests ==="
$PYTHON src/cg_eval_nondg_expB.py --arm nondg_enc_lora_s1337 --device cuda 2>&1 | tee eval_enc_lora_s1337_expB.log

echo "=== [SPARK1] Starting Training nondg_frozen_inst_s1337 (4,824 steps) ==="
$PYTHON src/cg_train_nondg_expB.py --arm nondg_frozen_inst_s1337 --epochs 3 --batch-size 8 --device cuda --eval-every 200 2>&1 | tee train_frozen_inst_s1337_expB.log

echo "=== [SPARK1] Evaluating nondg_frozen_inst_s1337 on all manifests ==="
$PYTHON src/cg_eval_nondg_expB.py --arm nondg_frozen_inst_s1337 --device cuda 2>&1 | tee eval_frozen_inst_s1337_expB.log

echo "=== [SPARK1] All Exp B jobs completed successfully! ==="
