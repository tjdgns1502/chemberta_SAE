#!/bin/bash

# Create log directory
mkdir -p /home/yoo122333/capstone/logs/sae_intervention

# GPU 0: Patterns 0-20 (21 patterns)
nohup /home/yoo122333/micromamba/envs/chemberta-repro/bin/python /home/yoo122333/capstone/chemberta_repro_final/docs/sae_intervention_experiment.py \
    --pattern_ids "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" \
    --gpu_id 0 \
    > /home/yoo122333/capstone/logs/sae_intervention/gpu0.log 2>&1 &

echo "Started GPU 0 (patterns 0-20)"

# GPU 3: Patterns 21-41 (21 patterns)
nohup /home/yoo122333/micromamba/envs/chemberta-repro/bin/python /home/yoo122333/capstone/chemberta_repro_final/docs/sae_intervention_experiment.py \
    --pattern_ids "21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41" \
    --gpu_id 3 \
    > /home/yoo122333/capstone/logs/sae_intervention/gpu3.log 2>&1 &

echo "Started GPU 3 (patterns 21-41)"

# GPU 4: Patterns 42-63 (22 patterns)
nohup /home/yoo122333/micromamba/envs/chemberta-repro/bin/python /home/yoo122333/capstone/chemberta_repro_final/docs/sae_intervention_experiment.py \
    --pattern_ids "42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" \
    --gpu_id 4 \
    > /home/yoo122333/capstone/logs/sae_intervention/gpu4.log 2>&1 &

echo "Started GPU 4 (patterns 42-63)"

echo ""
echo "All 64 intervention patterns started across 3 GPUs"
echo "Monitor progress with:"
echo "  tail -f /home/yoo122333/capstone/logs/sae_intervention/gpu0.log"
echo "  tail -f /home/yoo122333/capstone/logs/sae_intervention/gpu3.log"
echo "  tail -f /home/yoo122333/capstone/logs/sae_intervention/gpu4.log"
