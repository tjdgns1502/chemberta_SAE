#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yoo122333/micromamba/envs/chemberta-repro/bin/python3}"
CD="${CD:-/home/yoo122333/capstone/chemberta_SAE}"
SCRIPT="$CD/scripts/run_sae_probe.py"
WANDB_PROJECT="${WANDB_PROJECT:-sae_lens_training}"
N_LATENTS="${N_LATENTS:-1536}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-2}"

cd "$CD"

run_probe() {
    local gpu="$1"
    local layer="$2"
    local l0="$3"
    local run_id="$4"
    local run_name="$5"

    echo "[$(date '+%F %T')] START gpu=$gpu layer=$layer l0=$l0 run_id=$run_id"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT" \
        --layers "$layer" \
        --base-l0 "$l0" \
        --schedule none \
        --epochs 30 \
        --sparsity-loss-mode tanh \
        --n-latents "$N_LATENTS" \
        --run-id "$run_id" \
        --log-to-wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "$run_name"
    local rc=$?
    echo "[$(date '+%F %T')] END rc=$rc gpu=$gpu layer=$layer l0=$l0 run_id=$run_id"
    echo
    return "$rc"
}

worker() {
    local gpu="$1"
    shift

    while [ "$#" -gt 0 ]; do
        local layer="$1"
        local l0="$2"
        local run_id="$3"
        local run_name="$4"
        shift 4

        if ! run_probe "$gpu" "$layer" "$l0" "$run_id" "$run_name"; then
            echo "[$(date '+%F %T')] FAIL gpu=$gpu layer=$layer l0=$l0 run_id=$run_id"
        fi
    done
}

worker "$GPU_A" \
    1 8e-2  v4_tanh_n1536_L1_l0_8e2  v4_tanh_n1536_L1_l0_8e-2 \
    1 10e-2 v4_tanh_n1536_L1_l0_10e2 v4_tanh_n1536_L1_l0_10e-2 \
    1 12e-2 v4_tanh_n1536_L1_l0_12e2 v4_tanh_n1536_L1_l0_12e-2 \
    1 16e-2 v4_tanh_n1536_L1_l0_16e2 v4_tanh_n1536_L1_l0_16e-2 \
    1 20e-2 v4_tanh_n1536_L1_l0_20e2 v4_tanh_n1536_L1_l0_20e-2 &

worker "$GPU_B" \
    2 22e-2 v4_tanh_n1536_L2_l0_22e2 v4_tanh_n1536_L2_l0_22e-2 \
    2 24e-2 v4_tanh_n1536_L2_l0_24e2 v4_tanh_n1536_L2_l0_24e-2 \
    2 28e-2 v4_tanh_n1536_L2_l0_28e2 v4_tanh_n1536_L2_l0_28e-2 \
    2 32e-2 v4_tanh_n1536_L2_l0_32e2 v4_tanh_n1536_L2_l0_32e-2 &

worker "$GPU_C" \
    3 22e-2 v4_tanh_n1536_L3_l0_22e2 v4_tanh_n1536_L3_l0_22e-2 \
    4 20e-2 v4_tanh_n1536_L4_l0_20e2 v4_tanh_n1536_L4_l0_20e-2 \
    5 20e-2 v4_tanh_n1536_L5_l0_20e2 v4_tanh_n1536_L5_l0_20e-2 \
    3 24e-2 v4_tanh_n1536_L3_l0_24e2 v4_tanh_n1536_L3_l0_24e-2 \
    4 22e-2 v4_tanh_n1536_L4_l0_22e2 v4_tanh_n1536_L4_l0_22e-2 \
    5 22e-2 v4_tanh_n1536_L5_l0_22e2 v4_tanh_n1536_L5_l0_22e-2 &

wait
