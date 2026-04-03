#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yoo122333/micromamba/envs/chemberta-repro/bin/python3}"
CD="${CD:-/home/yoo122333/capstone/chemberta_SAE}"
SCRIPT="$CD/scripts/run_sae_probe.py"
WANDB_PROJECT="${WANDB_PROJECT:-sae_lens_training}"
N_LATENTS="${N_LATENTS:-1536}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
RUN_PREFIX="${RUN_PREFIX:-v5_tanh_n${N_LATENTS}}"

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
    1 11e-2 "${RUN_PREFIX}_L1_l0_11e2" "${RUN_PREFIX}_L1_l0_11e-2" \
    1 13e-2 "${RUN_PREFIX}_L1_l0_13e2" "${RUN_PREFIX}_L1_l0_13e-2" &

worker "$GPU_A" \
    1 14e-2 "${RUN_PREFIX}_L1_l0_14e2" "${RUN_PREFIX}_L1_l0_14e-2" \
    1 15e-2 "${RUN_PREFIX}_L1_l0_15e2" "${RUN_PREFIX}_L1_l0_15e-2" &

worker "$GPU_B" \
    2 13e-2 "${RUN_PREFIX}_L2_l0_13e2" "${RUN_PREFIX}_L2_l0_13e-2" \
    2 15e-2 "${RUN_PREFIX}_L2_l0_15e2" "${RUN_PREFIX}_L2_l0_15e-2" \
    2 18e-2 "${RUN_PREFIX}_L2_l0_18e2" "${RUN_PREFIX}_L2_l0_18e-2" &

worker "$GPU_B" \
    2 14e-2 "${RUN_PREFIX}_L2_l0_14e2" "${RUN_PREFIX}_L2_l0_14e-2" \
    2 17e-2 "${RUN_PREFIX}_L2_l0_17e2" "${RUN_PREFIX}_L2_l0_17e-2" &

wait
