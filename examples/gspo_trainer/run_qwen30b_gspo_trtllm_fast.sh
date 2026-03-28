# run Qwen3-30B GSPO with new model engine
set -x

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

NNODES=${NNODES:-4}

if [ ! -z "$EXPERIMENT_POSTFIX" ]; then
    EXPERIMENT_POSTFIX="-${EXPERIMENT_POSTFIX}"
fi

# wandb
backend=${backend:-megatron} # fsdp, fsdp2, megatron
project_name=${PROJECT_NAME:-'wuxibin_gspo'}

# ===================================== Algorithm =====================================
adv_estimator=grpo
loss_mode=gspo

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

clip_ratio_low=3e-4
clip_ratio_high=4e-4

actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

# ===================================== Data/Model =====================================
train_files=${train_files:-$DATA_ROOT/dataset/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet}
test_files=${test_files:-$DATA_ROOT/dataset/aime-2024.parquet}

actor_model_path=${actor_model_path:-$HDFS_ROOT/model/Qwen3-30B-A3B-Base}

MODEL=${MODEL:-30B}

critic_model_path=$actor_model_path

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

train_batch_size=256
ppo_mini_batch_size=32
n_resp_per_prompt=16
n_resp_per_prompt_val=1

MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-256}

# ===================================== Training =====================================
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 3))
critic_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4))

# FSDP parallelism config
USP_SIZE=4
ACTOR_FSDP_CONFIG="
    actor_rollout_ref.actor.fsdp_config.strategy=$backend \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$USP_SIZE" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16

# Megatron parallelism config
if [ "$MODEL" = "30B" ]; then
    train_vpp=2
    train_tp=4
    train_pp=2
    train_cp=1
    train_ep=8
    train_etp=1
    offload=True
    total_rollout_steps=1
    USE_MBRIDGE=True
    attention_backend=flash
elif [ "$MODEL" = "235B" ]; then
    train_vpp=1
    train_tp=4
    train_pp=8
    train_cp=2
    train_ep=8
    train_etp=1
    offload=True
    total_rollout_steps=1
    USE_MBRIDGE=True
    attention_backend=flash
fi
ACTOR_MEGATRON_CONFIG=(
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${train_vpp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=${attention_backend}
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="flex"
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True
    #+actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=10
    #+actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=True
    #+actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=True
    actor_rollout_ref.ref.megatron.param_offload=${offload}
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${train_vpp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
)


# Actor model config
ACTOR_CONFIG="
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu"

# Critic model config
CIRITC_CONFIG="
    critic.optim.lr=$critic_lr \
    critic.model.path=$critic_model_path \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
    critic.ulysses_sequence_parallel_size=$USP_SIZE"

CRITIC_FSDP_CONFIG="${ACTOR_FSDP_CONFIG//actor_rollout_ref.actor/critic.model}"
CRITIC_MEGATRON_CONFIG="${ACTOR_MEGATRON_CONFIG[@]//actor_rollout_ref.actor/critic}"

if [[ $backend == "megatron" ]]; then
    CONFIG_NAME=ppo_megatron_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG ${ACTOR_MEGATRON_CONFIG[@]}"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_MEGATRON_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
else # fsdp, fsdp2
    CONFIG_NAME=ppo_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_FSDP_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_FSDP_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
fi

# ===================================== Inference =====================================
rollout_name=trtllm
if [ "$rollout_name" = "vllm" ]; then
    export VLLM_USE_V1=1
fi
if [ "$MODEL" = "30B" ]; then
    infer_tp=4
    infer_dp=1
    infer_ep=4
    gpu_memory_utilization=0.7
elif [ "$MODEL" = "235B" ]; then
    infer_tp=8
    infer_dp=1
    infer_ep=4
    gpu_memory_utilization=0.8
fi

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_BATCH_SIZE} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=4096 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_timeout_iters=32 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_max_tokens_ratio=0.5 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val"

# ===================================== Reward =====================================
REWARD_CONFIG="
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}"

experiment_name=qwen3-${MODEL}-gspo-$backend-trtllm-n${NNODES}
default_local_dir=./checkpoint/$project_name/$experiment_name

python3 -m verl.trainer.main_ppo \
    --config-path=./config \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.truncation='error' \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=${experiment_name}${EXPERIMENT_POSTFIX} \
    trainer.default_local_dir=$default_local_dir \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.val_before_train=False \
    trainer.log_val_generations=100 \
    trainer.save_freq=5 \
    trainer.resume_mode=auto \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=500 \
    $ACTOR_CONFIG \
    $CIRITC_CONFIG \
    $ROLLOUT_CONFIG \
    $REWARD_CONFIG $@
