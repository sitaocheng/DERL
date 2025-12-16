
loop_num=$1
rollout_num=$2

export VLLM_WORKER_MULTIPROC_METHOD='spawn'

python ./eval_vllm.py \
    --model_path --model_path ./outputs/meta_grpo_dlc/merged_model/meta_llm_rl_from_sft/meta_outerloop${loop_num}_rollout${rollout_num}/ \
    --output_file_path meta_llm_outerloop${loop_num}_rollout${rollout_num}_test \
    --loop_num ${loop_num} \
    --rollout_num ${rollout_num} \
    --testing True
