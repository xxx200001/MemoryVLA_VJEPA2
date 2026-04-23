#!/bin/bash
set -euo pipefail
export MKL_INTERFACE_LAYER=GNU

# ============================================================
# MemoryVLA LIBERO-10 评估脚本
# 服务器: didi-02
# ============================================================

WORK_DIR="/data8/data/zhanght2504_didi2/runspace_xj/MemoryVLA-1"
cd "${WORK_DIR}"

# ---------- 要评估的 checkpoint 列表 ----------
ckpt_list=(
  "${WORK_DIR}/log/libero/memvla_libero_10--image_aug/checkpoints/step-004000-epoch-02-loss=0.0247.pt"
  # 如果有更多 checkpoint，继续添加:
  # "${WORK_DIR}/log/libero/memvla_libero_10--image_aug/checkpoints/step-008000-epoch-XX-loss=X.XXXX.pt"
)

# ---------- 评估参数 ----------
task_suite_name="libero_10"
gpu_id=0
action_chunking_window=8
num_trials_per_task=50
unnorm_key="libero_10_no_noops"

# ---------- 工具函数 ----------
find_free_port() {
  local min=${1:-2000}
  local max=${2:-30000}
  for ((i=0; i<1000; i++)); do
    port=$(shuf -i"${min}"-"${max}" -n1)
    if ! lsof -iTCP:"${port}" -sTCP:LISTEN &>/dev/null; then
      echo "${port}"
      return 0
    fi
  done
  echo "ERROR: no free port found" >&2
  return 1
}

export CUDA_VISIBLE_DEVICES=${gpu_id}

# ---------- 逐个 checkpoint 评估 ----------
for ckpt_path in "${ckpt_list[@]}"; do
  echo ""
  echo "========================================================"
  echo ">>> 评估 checkpoint: ${ckpt_path}"
  echo "========================================================"

  port=$(find_free_port)
  echo "    使用端口: ${port}"

  # 评估结果保存目录
  local_log_dir="$(dirname "$(dirname "$ckpt_path")")/eval_libero/$(basename "$ckpt_path")"
  mkdir -p "${local_log_dir}"
  echo "    结果目录: ${local_log_dir}"

  # === 第1步: 启动模型推理服务 ===
  echo "[1/3] 启动 deploy 推理服务..."
  python deploy.py \
    --saved_model_path "${ckpt_path}" \
    --unnorm_key "${unnorm_key}" \
    --adaptive_ensemble_alpha 0.1 \
    --cfg_scale 1.5 \
    --port "${port}" \
    --action_chunking \
    --action_chunking_window "${action_chunking_window}" &

  DEPLOY_PID=$!
  echo "    deploy PID: ${DEPLOY_PID}"

  # === 第2步: 等待模型加载 ===
  # MemoryVLA ~7B 参数, 加载需要较长时间
  # 4 GPU 训练的模型用单卡推理, 大约需要 10-30 分钟
  echo "[2/3] 等待模型加载 (20 分钟)..."
  echo "    如果模型已提前加载完毕, 可以 Ctrl+C 后手动跑 eval 命令"
  sleep 1200

  # === 第3步: 跑仿真评估 ===
  echo "[3/3] 开始仿真评估 (task_suite=${task_suite_name}, trials=${num_trials_per_task})..."
  python evaluation/libero/eval_libero.py \
    --task_suite_name "${task_suite_name}" \
    --num_trials_per_task "${num_trials_per_task}" \
    --run_id_note "ac${action_chunking_window}" \
    --local_log_dir "${local_log_dir}" \
    --port "${port}"

  # 清理
  echo "    杀掉 deploy 进程 (PID ${DEPLOY_PID})"
  kill "${DEPLOY_PID}" 2>/dev/null || true
  wait "${DEPLOY_PID}" 2>/dev/null || true

  echo ">>> 完成: ${ckpt_path}"
  echo ""
done

echo "========================================================"
echo "全部评估完成!"
echo ""
echo "查看结果:"
echo "  grep 'Current total success rate' ${WORK_DIR}/log/libero/memvla_libero_10--image_aug/eval_libero/*/*.txt"
echo "========================================================"
