#!/bin/bash
# Run queued tasks on every node of this job until the queue is empty.
#
# The queue is a folder (MIDTRAIN_QUEUE) with four subfolders: pending, running, done, failed. A task is one
# executable shell script. A node claims a task by renaming it from pending/ to running/ (a rename is atomic on the
# shared file system, so two nodes never run the same task) and moves it to done/ or failed/ afterwards, with its
# output next to it. Two kinds of tasks, told apart by their file name:
#   node__<name>.sh   needs every GPU of one node (a 9B run); it sees all of them
#   gpu__<name>.sh    needs one GPU (a proxy run); a node runs one such task per GPU at a time
# A node takes whole-node tasks first. When no task of either kind is left it waits MIDTRAIN_WORKER_IDLE_MINUTES
# (default 0) for new ones and then stops; the job ends when every node has stopped. The wait exists because a job
# that starts earlier than expected would otherwise leave at once and give up a place it queued days for.
# MIDTRAIN_WORKER_GPUS (for example "1,2,3") limits the GPUs used, for a workstation shared with other jobs.
#
# Run by scripts/cluster/worker.slurm. To try it on one machine:
#   MIDTRAIN_QUEUE=/path/to/queue bash scripts/cluster/node_worker.sh --this-node-only
set -uo pipefail
queue_directory="${MIDTRAIN_QUEUE:?set MIDTRAIN_QUEUE to the queue folder}"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${queue_directory}"/{pending,running,done,failed}

claim_next_task() {   # claim_next_task <prefix> → prints the claimed path in running/, or nothing
  local prefix="$1" candidate claimed
  for candidate in $(ls "${queue_directory}/pending" 2>/dev/null | grep "^${prefix}__" | sort); do
    claimed="${queue_directory}/running/${candidate%.sh}.$(hostname -s).sh"
    if mv "${queue_directory}/pending/${candidate}" "${claimed}" 2>/dev/null; then
      echo "${claimed}"
      return 0
    fi
  done
  return 1
}

run_claimed_task() {   # run_claimed_task <path in running/> <CUDA_VISIBLE_DEVICES>
  local task_path="$1" visible_gpus="$2" task_name status
  task_name="$(basename "${task_path}" .sh)"
  echo "$(date '+%m-%d %H:%M') $(hostname -s) gpus ${visible_gpus}: start ${task_name}"
  CUDA_VISIBLE_DEVICES="${visible_gpus}" bash "${task_path}" > "${queue_directory}/running/${task_name}.out" 2>&1
  status=$?
  if [[ ${status} -eq 0 ]]; then
    mv "${task_path}" "${queue_directory}/running/${task_name}.out" "${queue_directory}/done/"
  else
    mv "${task_path}" "${queue_directory}/running/${task_name}.out" "${queue_directory}/failed/"
  fi
  echo "$(date '+%m-%d %H:%M') $(hostname -s) gpus ${visible_gpus}: ${task_name} exit ${status}"
}

pending_task_count() { ls "${queue_directory}/pending" 2>/dev/null | grep -c -E '^(node|gpu)__.*\.sh$'; }

work_on_this_node() {
  local all_gpus task_path gpu_index idle_seconds_left
  all_gpus="${MIDTRAIN_WORKER_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)}"
  [[ -n "${all_gpus}" ]] || { echo "$(hostname -s): no GPU visible"; return 1; }
  idle_seconds_left=$(( ${MIDTRAIN_WORKER_IDLE_MINUTES:-0} * 60 ))
  while true; do
    while task_path="$(claim_next_task node)"; do
      run_claimed_task "${task_path}" "${all_gpus}"
    done
    for gpu_index in ${all_gpus//,/ }; do
      ( while task_path="$(claim_next_task gpu)"; do run_claimed_task "${task_path}" "${gpu_index}"; done ) &
    done
    wait
    # nothing left: wait for new tasks in 30-second looks, up to the idle allowance, then stop
    while [[ "$(pending_task_count)" -eq 0 && ${idle_seconds_left} -gt 0 ]]; do
      sleep 30
      idle_seconds_left=$(( idle_seconds_left - 30 ))
    done
    [[ "$(pending_task_count)" -gt 0 ]] || break
  done
}

if [[ "${1:-}" == "--this-node-only" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
  work_on_this_node
else
  # one worker per node, started over ssh with the queue location passed along; this shell waits for all of them
  for node_name in $(scontrol show hostnames "${SLURM_JOB_NODELIST}"); do
    ssh -o StrictHostKeyChecking=no "${node_name}" \
      "MIDTRAIN_QUEUE='${queue_directory}' MIDTRAIN_WORKER_IDLE_MINUTES='${MIDTRAIN_WORKER_IDLE_MINUTES:-0}' \
       bash '${script_directory}/node_worker.sh' --this-node-only" &
  done
  wait
fi
echo "$(date '+%m-%d %H:%M') worker finished: $(ls "${queue_directory}/done" | grep -c '\.sh$') done, $(ls "${queue_directory}/failed" | grep -c '\.sh$') failed, $(ls "${queue_directory}/pending" | grep -c '\.sh$') pending"
