#!/bin/bash
# Copy code and data to the cluster and bring result files back.
#
# Why. The workstation and the cluster run the same tasks from the same code, and the cluster nodes have no
# network. Everything a task needs therefore has to be copied there first, and every result file has to come back
# before it can be committed. Result files are named after their run, and a run happens on one machine, so the
# two sides never write the same file: results are copied both ways without deleting anything.
#
# Settings, in site.env on the workstation:
#   MIDTRAIN_CLUSTER_HOST         the ssh host name of the cluster
#   MIDTRAIN_CLUSTER_RSYNC_SHELL  the remote shell for rsync, for example "ssh -o ControlPath=..." (default: ssh)
#   MIDTRAIN_CLUSTER_ROOT         the repository on the cluster
#   MIDTRAIN_CLUSTER_DATA         MIDTRAIN_DATA on the cluster
#   MIDTRAIN_CLUSTER_MODELS       MIDTRAIN_MODELS on the cluster
#
#   bash scripts/cluster/sync.sh code               # code and the small committed result files
#   bash scripts/cluster/sync.sh data               # token files, evaluation items, data of both probes, gate file, packs, subsets
#   bash scripts/cluster/sync.sh model 2B           # one base checkpoint
#   bash scripts/cluster/sync.sh results            # result files written on the cluster, back into results/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cluster_host="${MIDTRAIN_CLUSTER_HOST:?set MIDTRAIN_CLUSTER_HOST in site.env}"
remote_shell="${MIDTRAIN_CLUSTER_RSYNC_SHELL:-ssh}"

case "${1:-}" in
  code)
    # --delete keeps the cluster copy equal to this one; results/ and site.env are left alone on the other side
    rsync -a --delete -e "${remote_shell}" --exclude .git --exclude site.env --exclude /results --exclude __pycache__ \
      --exclude .pytest_cache --exclude /midtrain_data "${repository_root}/" "${cluster_host}:${MIDTRAIN_CLUSTER_ROOT}/"
    rsync -a -e "${remote_shell}" "${repository_root}/results/" "${cluster_host}:${MIDTRAIN_CLUSTER_ROOT}/results/"
    ;;
  data)
    for part in tokens eval probe probe_generic hygiene packs subsets; do
      [[ -d "${MIDTRAIN_DATA}/${part}" ]] || continue
      rsync -a --partial -e "${remote_shell}" "${MIDTRAIN_DATA}/${part}/" "${cluster_host}:${MIDTRAIN_CLUSTER_DATA}/${part}/"
      echo "$(date '+%H:%M') copied ${part}"
    done
    ;;
  model)
    model_folder="qwen35_${2:?give a model size, for example 2B}_base"
    rsync -a --partial -e "${remote_shell}" "${MIDTRAIN_MODELS}/${model_folder}/" \
      "${cluster_host}:${MIDTRAIN_CLUSTER_MODELS}/${model_folder}/"
    echo "$(date '+%H:%M') copied ${model_folder}"
    ;;
  results)
    rsync -a -e "${remote_shell}" "${cluster_host}:${MIDTRAIN_CLUSTER_ROOT}/results/" "${repository_root}/results/"
    ;;
  *)
    echo "usage: sync.sh code | data | model <size> | results" >&2
    exit 2
    ;;
esac
