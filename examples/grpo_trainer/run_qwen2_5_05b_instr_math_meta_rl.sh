set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

gsm8k_train_path=./dataset/gsm8k/train.parquet
gsm8k_test_path=./dataset/gsm8k/test.parquet
math_train_path=./dataset/math/train.parquet
math_test_path=./dataset/math/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"


export VLLM_WORKER_MULTIPROC_METHOD='spawn'
export HYDRA_FULL_ERROR=1

export VERL_ROOT_DIR="./verl"
export PYTHONPATH=${VERL_ROOT_DIR}:${PYTHONPATH}

new_iter=$1
model_path=$2
previous_iter=$3

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_epochs=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_gsm8k_math_meta_llm_rl_from_sft_outer_loop_dlc' \
    trainer.experiment_name='qwen2.5-05b-outer_loop_'${new_iter} \
    trainer.n_gpus_per_node=4 \
    +outer_iter=$previous_iter \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    trainer.outer='outer' \
    +custom_reward='1'
