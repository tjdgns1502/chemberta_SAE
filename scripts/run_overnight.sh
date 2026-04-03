#!/usr/bin/env bash
# Overnight experiment runner — tanh mode layer sweep + n_latents comparison
# Run inside tmux: tmux new -s sae && bash scripts/run_overnight.sh
set -euo pipefail

PYTHON=/home/yoo122333/micromamba/envs/chemberta-repro/bin/python3
CD=/home/yoo122333/capstone/chemberta_SAE
SCRIPT="$CD/scripts/run_sae_probe.py"
WANDB_PROJECT="sae_lens_training"

cd "$CD"

run_probe() {
    local gpu=$1 layer=$2 n_latents=$3 l0=$4 run_id=$5 run_name=$6
    echo "[$(date '+%H:%M:%S')] START  GPU=$gpu  $run_name"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$SCRIPT" \
        --layers "$layer" --base-l0 "$l0" --schedule none --epochs 30 \
        --sparsity-loss-mode tanh --n-latents "$n_latents" \
        --run-id "$run_id" \
        --log-to-wandb --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "$run_name" \
        2>&1 | tail -5
    echo "[$(date '+%H:%M:%S')] DONE   $run_name"
    echo ""
}

echo "============================================"
echo " Overnight SAE Experiments"
echo " Started: $(date)"
echo "============================================"
echo ""

# ============================================================
# ROUND 1: Layer sweep with l0=1e-2 (gentler for deeper layers)
# + Layer 5 with l0=5e-2 to complete the survey
# ============================================================
echo "=== ROUND 1: Layer sweep l0=1e-2 + Layer 5 survey ==="

run_probe 0 5 1536 5e-2 "v2_tanh_n1536_layer5"      "v2_tanh_n1536_layer5_l0_5e-2" &
run_probe 1 1 1536 1e-2 "v2_tanh_n1536_L1_l0_1e2"   "v2_tanh_n1536_L1_l0_1e-2" &
run_probe 2 2 1536 1e-2 "v2_tanh_n1536_L2_l0_1e2"   "v2_tanh_n1536_L2_l0_1e-2" &
run_probe 3 3 1536 1e-2 "v2_tanh_n1536_L3_l0_1e2"   "v2_tanh_n1536_L3_l0_1e-2" &
run_probe 4 4 1536 1e-2 "v2_tanh_n1536_L4_l0_1e2"   "v2_tanh_n1536_L4_l0_1e-2" &
run_probe 5 5 1536 1e-2 "v2_tanh_n1536_L5_l0_1e2"   "v2_tanh_n1536_L5_l0_1e-2" &
wait
echo "=== ROUND 1 COMPLETE ==="
echo ""

# ============================================================
# ROUND 2: Mid-range l0=2e-2 for layers 1-5
# ============================================================
echo "=== ROUND 2: Layer sweep l0=2e-2 ==="

run_probe 0 1 1536 2e-2 "v2_tanh_n1536_L1_l0_2e2"   "v2_tanh_n1536_L1_l0_2e-2" &
run_probe 1 2 1536 2e-2 "v2_tanh_n1536_L2_l0_2e2"   "v2_tanh_n1536_L2_l0_2e-2" &
run_probe 2 3 1536 2e-2 "v2_tanh_n1536_L3_l0_2e2"   "v2_tanh_n1536_L3_l0_2e-2" &
run_probe 3 4 1536 2e-2 "v2_tanh_n1536_L4_l0_2e2"   "v2_tanh_n1536_L4_l0_2e-2" &
run_probe 4 5 1536 2e-2 "v2_tanh_n1536_L5_l0_2e2"   "v2_tanh_n1536_L5_l0_2e-2" &
# GPU 5: n=2048 Layer 0 l0=3e-2 (fill the Pareto curve)
run_probe 5 0 2048 1e-2 "v2_tanh_n2048_L0_l0_1e2"   "v2_tanh_n2048_L0_l0_1e-2" &
wait
echo "=== ROUND 2 COMPLETE ==="
echo ""

# ============================================================
# ROUND 3: n=2048 for all layers at best l0 per layer
# Using l0=3e-2 as a good middle ground for n=2048
# ============================================================
echo "=== ROUND 3: n=2048 all layers l0=3e-2 ==="

run_probe 0 1 2048 3e-2 "v2_tanh_n2048_L1_l0_3e2"   "v2_tanh_n2048_L1_l0_3e-2" &
run_probe 1 2 2048 3e-2 "v2_tanh_n2048_L2_l0_3e2"   "v2_tanh_n2048_L2_l0_3e-2" &
run_probe 2 3 2048 3e-2 "v2_tanh_n2048_L3_l0_3e2"   "v2_tanh_n2048_L3_l0_3e-2" &
run_probe 3 4 2048 3e-2 "v2_tanh_n2048_L4_l0_3e2"   "v2_tanh_n2048_L4_l0_3e-2" &
run_probe 4 5 2048 3e-2 "v2_tanh_n2048_L5_l0_3e2"   "v2_tanh_n2048_L5_l0_3e-2" &
# GPU 5: n=2048 Layer 0 with l0=1e-1 (strong sparsity)
run_probe 5 0 2048 1e-1 "v2_tanh_n2048_L0_l0_1e1"   "v2_tanh_n2048_L0_l0_1e-1" &
wait
echo "=== ROUND 3 COMPLETE ==="
echo ""

# ============================================================
# ROUND 4: n=2048 layers with l0=1e-2 (gentle) for comparison
# ============================================================
echo "=== ROUND 4: n=2048 all layers l0=1e-2 ==="

run_probe 0 1 2048 1e-2 "v2_tanh_n2048_L1_l0_1e2"   "v2_tanh_n2048_L1_l0_1e-2" &
run_probe 1 2 2048 1e-2 "v2_tanh_n2048_L2_l0_1e2"   "v2_tanh_n2048_L2_l0_1e-2" &
run_probe 2 3 2048 1e-2 "v2_tanh_n2048_L3_l0_1e2"   "v2_tanh_n2048_L3_l0_1e-2" &
run_probe 3 4 2048 1e-2 "v2_tanh_n2048_L4_l0_1e2"   "v2_tanh_n2048_L4_l0_1e-2" &
run_probe 4 5 2048 1e-2 "v2_tanh_n2048_L5_l0_1e2"   "v2_tanh_n2048_L5_l0_1e-2" &
wait
echo "=== ROUND 4 COMPLETE ==="
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " Finished: $(date)"
echo "============================================"
echo ""
echo "Results summary:"
$PYTHON -c "
import torch, glob, json, os

base = '/home/yoo122333/capstone/chemberta_SAE/artifacts/runs/sae'
results = []

for d in sorted(glob.glob(f'{base}/v2_tanh_*')):
    run_id = os.path.basename(d)
    try:
        meta = json.load(open(f'{d}/reports/wandb_run.json'))
        name = meta.get('wandb_run_name', run_id)
    except:
        name = run_id

    for layer_dir in sorted(glob.glob(f'{d}/probe/checkpoints/layer_*')):
        layer = int(layer_dir.split('_')[-1])
        best_path = f'{layer_dir}/best.pt'
        if not os.path.exists(best_path):
            continue
        ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
        q = ckpt.get('quality', {})
        results.append({
            'name': name,
            'layer': layer,
            'nmse': ckpt.get('best_nmse', 0),
            'l0': q.get('mean_l0', 0),
            'dead': q.get('dead_ratio', 0),
        })

print(f'{\"name\":45s} {\"L\":>2s} {\"NMSE\":>10s} {\"L0\":>7s} {\"dead\":>7s}')
print('-' * 75)
for r in sorted(results, key=lambda x: (x['layer'], x['dead'], x['nmse'])):
    n_est = 2048 if 'n2048' in r['name'] else 1536
    pct = r['l0'] / n_est * 100
    print(f'{r[\"name\"]:45s} {r[\"layer\"]:2d} {r[\"nmse\"]:10.6f} {r[\"l0\"]:6.1f}({pct:4.1f}%) {r[\"dead\"]:7.4f}')
"
