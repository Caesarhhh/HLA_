#!/usr/bin/env bash
set -euo pipefail

# Common direct-node launcher for matched GDN/HLA 1.3B pretraining.
# Do not invoke this file directly; use one of the variant wrappers beside it.
# This script does not submit, query, or modify any Slurm job.

if [[ "${MODEL_VARIANT:-}" != "gdn" && "${MODEL_VARIANT:-}" != "hla" ]]; then
  echo "MODEL_VARIANT must be set to gdn or hla by a wrapper." >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TRAIN_DATA="${TRAIN_DATA:-${REPO}/data/slimpajama627b_gdn_packed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/experiments/gdn_hla_1p3b_100b_from_scratch_20260911}"
RUN_TAG="${RUN_TAG:-s3407_v1}"

TRAIN_CONFIG="${TRAIN_CONFIG:-512x4k_100B}"
SEQ_LEN=4096
GLOBAL_BATCH_SEQUENCES="${GLOBAL_BATCH_SEQUENCES:-512}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$("$PYTHON_BIN" -c 'import torch; print(",".join([str(i) for i in range(torch.cuda.device_count())]))')
  [[ -n "$CUDA_VISIBLE_DEVICES" ]] || { echo "No visible CUDA GPU." >&2; exit 1; }
fi

# 47,684 complete optimizer steps * 512 sequences * 4,096 tokens.
# This is 100B rounded up by 595,968 tokens (0.000596%) and avoids a final
# partial gradient-accumulation window in pretrain.py.
MAX_TOKENS="${MAX_TOKENS:-100000595968}"
WARMUP_TOKENS="${WARMUP_TOKENS:-1000000000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-1}"
SEED="${SEED:-3407}"
SAVE_STEP_INTERVAL="${SAVE_STEP_INTERVAL:-250}"
SAVE_STEP_ARCHIVE_INTERVAL="${SAVE_STEP_ARCHIVE_INTERVAL:-10000}"
ROUTER_STATS_STEP_INTERVAL="${ROUTER_STATS_STEP_INTERVAL:-25}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"

# The HLA architecture and auxiliary losses are fixed to the paper recipe.
case "$MODEL_VARIANT" in
  gdn)
    MODEL_NAME=GatedDeltaNet_Release_1.3B
    EXP_NAME="${EXP_NAME:-gdn_1p3b_scratch_slimpajama100pct_4k_100b_${RUN_TAG}}"
    ;;
  hla)
    MODEL_NAME=GatedDeltaNet_GLA_GDN_Release_1.3B
    EXP_NAME="${EXP_NAME:-hla_1p3b_scratch_affineab_p16_c256_slimpajama100pct_4k_100b_${RUN_TAG}}"
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$REPO/pretrain.py" ]]; then
  echo "Missing training entrypoint: $REPO/pretrain.py" >&2
  exit 1
fi
if [[ ! -d "$TRAIN_DATA" ]]; then
  echo "Missing training data: $TRAIN_DATA" >&2
  exit 1
fi
if ! compgen -G "$TRAIN_DATA/train_slim*" >/dev/null; then
  echo "No train_slim shards found in: $TRAIN_DATA" >&2
  exit 1
fi
if [[ -n "${TRAIN_DATA_EXTRA:-}" ]]; then
  echo "TRAIN_DATA_EXTRA is intentionally unsupported: this run must use 100% original data." >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=${#GPU_IDS[@]}
if (( GPU_COUNT < 1 )); then
  echo "CUDA_VISIBLE_DEVICES does not contain a GPU." >&2
  exit 1
fi
DENOM=$((MICRO_BATCH_SIZE * GPU_COUNT))
if (( GLOBAL_BATCH_SEQUENCES % DENOM != 0 )); then
  echo "GLOBAL_BATCH_SEQUENCES=$GLOBAL_BATCH_SEQUENCES is not divisible by micro_batch=$MICRO_BATCH_SIZE * GPUs=$GPU_COUNT." >&2
  exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SEQUENCES / DENOM))
TOKENS_PER_STEP=$((GLOBAL_BATCH_SEQUENCES * SEQ_LEN))
if (( MAX_TOKENS % TOKENS_PER_STEP != 0 )); then
  echo "MAX_TOKENS=$MAX_TOKENS must be divisible by tokens_per_step=$TOKENS_PER_STEP." >&2
  exit 1
fi
OPTIMIZER_STEPS=$((MAX_TOKENS / TOKENS_PER_STEP))

# Checkpoints are irreplaceable: only allow durable shared storage.
mkdir -p "$OUTPUT_ROOT"
OUTPUT_REAL=$(readlink -f "$OUTPUT_ROOT")
case "$OUTPUT_REAL" in
  /tmp/*|/local/*|/var/tmp/*)
    echo "Refusing node-local checkpoint root: $OUTPUT_REAL" >&2
    exit 1
    ;;
esac
WRITE_PROBE="$OUTPUT_ROOT/.checkpoint_write_probe.$$"
printf 'shared-checkpoint-probe\n' > "$WRITE_PROBE"
test -s "$WRITE_PROBE"
rm -f "$WRITE_PROBE"

OUT_DIR="$OUTPUT_ROOT/outputs/${TRAIN_CONFIG}_${EXP_NAME}"
if [[ -s "$OUT_DIR/final-model-ckpt.pth" ]]; then
  echo "Run is already complete: $OUT_DIR/final-model-ckpt.pth" >&2
  exit 1
fi
if [[ -d "$OUT_DIR" && ! -s "$OUT_DIR/latest-model-ckpt.pth" ]]; then
    echo "Output directory exists without a resumable latest checkpoint: $OUT_DIR" >&2
  exit 1
fi

LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_${EXP_NAME}.log}"

# Caches are disposable and may live on node-local storage; checkpoints never do.
CACHE_TAG="gdn_hla_1p3b_100b_${MODEL_VARIANT}_${RUN_TAG//[^a-zA-Z0-9_.-]/_}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-/tmp/${USER:-user}_${CACHE_TAG}_wandb}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER:-user}_${CACHE_TAG}_triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/${USER:-user}_${CACHE_TAG}_inductor}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/${USER:-user}_${CACHE_TAG}_pycache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/${USER:-user}_${CACHE_TAG}_xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-user}_${CACHE_TAG}_mpl}"
export TMPDIR="${TMPDIR:-/tmp/${USER:-user}_${CACHE_TAG}_tmp}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$WANDB_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"

COMMON_ARGS=(
  --train_data_dir "$TRAIN_DATA"
  --output_root "$OUTPUT_ROOT"
  --exp_name "$EXP_NAME"
  --exp_group gdn_hla_1p3b_100b_from_scratch
  --model_name "$MODEL_NAME"
  --train_config "$TRAIN_CONFIG"
  --learning_rate "$LEARNING_RATE"
  --weight_decay "$WEIGHT_DECAY"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --gradient_accumulation_steps_override "$GRADIENT_ACCUMULATION_STEPS"
  --max_tokens_override "$MAX_TOKENS"
  --warmup_tokens_override "$WARMUP_TOKENS"
  --save_step_checkpoints
  --save_step_interval "$SAVE_STEP_INTERVAL"
  --save_step_archive_interval "$SAVE_STEP_ARCHIVE_INTERVAL"
  --eval_step_interval 1000
  --eval_iters 10
  --log_step_interval 1
  --train_num_workers "$TRAIN_NUM_WORKERS"
  --seed "$SEED"
  --interactive_job
)
if [[ "${STOP_AFTER_STEP:-0}" != 0 ]]; then
  COMMON_ARGS+=(--stop_after_step "$STOP_AFTER_STEP")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  COMMON_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" --resume_dataloader_from_counters)
fi
if [[ -n "${RESUME_MICRO_BATCH_SIZE:-}" ]]; then
  # This is the saved checkpoint's micro batch, used to preserve consumed data
  # and token counters when the new MICRO_BATCH_SIZE differs.
  COMMON_ARGS+=(--resume_micro_batch_size "$RESUME_MICRO_BATCH_SIZE")
fi

HLA_ARGS=()
if [[ "$MODEL_VARIANT" == hla ]]; then
  export GDN_COMPILE_AFFINE_HISTORY_TRAINING="${GDN_COMPILE_AFFINE_HISTORY_TRAINING:-1}"
  HLA_ARGS=(
    --router_stats_step_interval "$ROUTER_STATS_STEP_INTERVAL"
    --router_stats_first_step
  )
fi

echo "[$(date -Is)] variant=$MODEL_VARIANT model=$MODEL_NAME init=random-from-scratch trainable=all"
echo "[$(date -Is)] data=$TRAIN_DATA mix=100% original SlimPajama; no extra/code-heavy source"
echo "[$(date -Is)] token_budget=$MAX_TOKENS optimizer_steps=$OPTIMIZER_STEPS sequence_length=$SEQ_LEN"
echo "[$(date -Is)] GPUs=$GPU_COUNT micro_batch=$MICRO_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS global_batch_sequences=$GLOBAL_BATCH_SEQUENCES tokens_per_step=$TOKENS_PER_STEP"
echo "[$(date -Is)] peak_lr=$LEARNING_RATE min_lr=$("$PYTHON_BIN" -c "print($LEARNING_RATE / 10)") warmup_tokens=$WARMUP_TOKENS weight_decay=$WEIGHT_DECAY"
echo "[$(date -Is)] checkpoints=$OUT_DIR  archives_every=$SAVE_STEP_ARCHIVE_INTERVAL"
if [[ "$MODEL_VARIANT" == hla ]]; then
  echo "[$(date -Is)] HLA C256/P16 self-attention pooling, affine sigmoid bias=2.2; losses=0.005/0.001/0.001"
fi

CMD=("$PYTHON_BIN" -u "$REPO/pretrain.py" "${COMMON_ARGS[@]}" "${HLA_ARGS[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY RUN command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "$REPO"
env \
  -u SLURM_NTASKS -u SLURM_NTASKS_PER_NODE -u SLURM_TASKS_PER_NODE \
  -u SLURM_PROCID -u SLURM_LOCALID -u SLURM_NODEID \
  -u SLURM_STEP_ID -u SLURM_STEP_NUM_TASKS \
  SLURM_NNODES=1 \
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
