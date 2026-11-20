#!/bin/bash
# A worker job for a Slurm cluster: it does nothing by itself, it runs whatever tasks are in the queue folder when
# it starts.
#
# Why. On a busy shared cluster a job waits days before it starts, and what it should run is rarely known when it
# is submitted. This script is only a pointer: Slurm copies it at submission, and it hands over to
# node_worker.sh, which is read when the job starts and can be rewritten until then. Tasks are added to the queue
# folder at any time (scripts/cluster/README.md). A job that finds no task exits within a minute.
#
# Account, partition, node count and time limit are site-specific and are given on the command line:
#   sbatch -A <account> -p <partition> -N 8 -n 8 -t 48:00:00 \
#          --export=ALL,MIDTRAIN_CLUSTER_ROOT=<repository on the cluster>,MIDTRAIN_QUEUE=<queue folder> \
#          scripts/cluster/worker.slurm
#SBATCH -J midtrain-worker
#SBATCH -o midtrain-worker-%j.out
set -uo pipefail
exec bash "${MIDTRAIN_CLUSTER_ROOT:?set MIDTRAIN_CLUSTER_ROOT to the repository on the cluster}/scripts/cluster/node_worker.sh"
