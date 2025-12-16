set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

gsm8k_train_path=/mnt/shared-storage-user/chengsitao/projects/dataset/gsm8k/train.parquet
gsm8k_test_path=/mnt/shared-storage-user/chengsitao/projects/dataset/gsm8k/test.parquet
math_train_path=/mnt/shared-storage-user/chengsitao/projects/dataset/math/train.parquet
math_test_path=/mnt/shared-storage-user/chengsitao/projects/dataset/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"


export VLLM_WORKER_MULTIPROC_METHOD='spawn'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export https_proxy=https://chengsitao:ppe6C7alaR93Uy6aipptd2HG1N1NfRa95qLMFSNN8thDqt4s0iY3ckY0ZRDe@aliyun-proxy.pjlab.org.cn:13128
export http_proxy=https://chengsitao:ppe6C7alaR93Uy6aipptd2HG1N1NfRa95qLMFSNN8thDqt4s0iY3ckY0ZRDe@aliyun-proxy.pjlab.org.cn:13128
export HTTP_PROXY=https://chengsitao:ppe6C7alaR93Uy6aipptd2HG1N1NfRa95qLMFSNN8thDqt4s0iY3ckY0ZRDe@aliyun-proxy.pjlab.org.cn:13128
export HTTPS_PROXY=https://chengsitao:ppe6C7alaR93Uy6aipptd2HG1N1NfRa95qLMFSNN8thDqt4s0iY3ckY0ZRDe@aliyun-proxy.pjlab.org.cn:13128


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/chengsitao/model/Qwen2.5-3B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='qwen2.5-3b-base_function_rule_rm_math' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 $@