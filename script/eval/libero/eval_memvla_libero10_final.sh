#!/bin/bash
# ============================================================
# MemoryVLA LIBERO-10 正式评估脚本
# 训练完成后使用，一键跑完所有 checkpoint
# ============================================================
set -euo pipefail

WORK_DIR="/data8/data/zhanght2504_didi2/runspace_xj/MemoryVLA-1"
cd "${WORK_DIR}"

# ----- 环境设置（已验证可用）-----
export LD_LIBRARY_PATH=~/osmesa_local/extracted/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${WORK_DIR}/third_libs/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa

# ----- 要评估的 checkpoint（训练完成后修改这里）-----
CKPT_DIR="${WORK_DIR}/log/libero/memvla_libero_10--image_aug/checkpoints"
ckpt_list=(
  # 建议评估最后几个 checkpoint，取最好的
  "${CKPT_DIR}/step-018000-epoch-*"
  "${CKPT_DIR}/step-019000-epoch-*"
  "${CKPT_DIR}/step-020000-epoch-*"
)

# ----- 评估参数 -----
task_suite_name="libero_10"
gpu_id=0              # 训练完成后 GPU 0 就空了
unnorm_key="libero_10_no_noops"
action_chunking_window=8
num_trials_per_task=50
port=6800

# ----- 开始评估 -----
for ckpt_pattern in "${ckpt_list[@]}"; do
  # 展开通配符
  ckpt_path=$(ls -1 ${ckpt_pattern} 2>/dev/null | head -1)
  if [ -z "${ckpt_path}" ]; then
    echo "⚠️  未找到: ${ckpt_pattern}, 跳过"
    continue
  fi

  ckpt_name=$(basename "${ckpt_path}" .pt)
  log_dir="${WORK_DIR}/log/libero/memvla_libero_10--image_aug/eval_libero/${ckpt_name}"
  mkdir -p "${log_dir}"

  echo ""
  echo "========================================================"
  echo ">>> 评估: ${ckpt_name}"
  echo ">>> 结果: ${log_dir}"
  echo "========================================================"

  # 启动推理服务
  CUDA_VISIBLE_DEVICES=${gpu_id} python deploy.py \
    --saved_model_path "${ckpt_path}" \
    --unnorm_key "${unnorm_key}" \
    --adaptive_ensemble_alpha 0.1 \
    --cfg_scale 1.5 \
    --port ${port} \
    --action_chunking \
    --action_chunking_window ${action_chunking_window} \
    --use_bf16 &
  DEPLOY_PID=$!

  # 等待模型加载（检测端口就绪）
  echo "等待模型加载..."
  for i in $(seq 1 60); do
    if curl -s http://localhost:${port}/ >/dev/null 2>&1; then
      echo "✅ 模型就绪！(${i}x30s)"
      break
    fi
    sleep 30
    echo "  加载中... ${i}x30s"
  done

  # 跑评估
  python evaluation/libero/eval_libero.py \
    --task_suite_name "${task_suite_name}" \
    --num_trials_per_task ${num_trials_per_task} \
    --run_id_note "ac${action_chunking_window}" \
    --local_log_dir "${log_dir}" \
    --port ${port}

  # 清理
  kill ${DEPLOY_PID} 2>/dev/null || true
  wait ${DEPLOY_PID} 2>/dev/null || true
  sleep 10

  echo ">>> 完成: ${ckpt_name}"
done

# ----- 汇总结果 -----
echo ""
echo "========================================================"
echo "📊 所有结果汇总:"
echo "========================================================"
for f in ${WORK_DIR}/log/libero/memvla_libero_10--image_aug/eval_libero/*/libero_10*.txt; do
  result=$(grep "Current total success rate" "$f" | tail -1)
  if [ -n "${result}" ]; then
    echo "  $(basename $(dirname $f)): ${result}"
  fi
done
