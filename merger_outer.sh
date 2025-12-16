loop_num=$1

python ./scripts/model_merger.py merge    \
    --backend fsdp    \
    --local_dir ./checkpoints/verl_grpo_gsm8k_math_meta_llm_rl_from_sft_outer_loop_dlc/qwen2.5-05b-outer_loop_${loop_num}/global_step_1/actor    \
    --target_dir ./outputs/meta_grpo_dlc/merged_model/meta_model/outer_loop_${loop_num}
