loop_num=$1
rollout_num=$2
checkpoints=$3

python ./scripts/model_merger.py merge    \
    --backend fsdp    \
    --local_dir $checkpoints   \
    --target_dir ./outputs/meta_grpo_dlc/merged_model/meta_llm_rl_from_sft/meta_outerloop${loop_num}_rollout${rollout_num}