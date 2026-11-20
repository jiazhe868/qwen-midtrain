#!/bin/bash
# Source this from every script: it loads the machine-specific paths (site.env) and puts the code on PYTHONPATH.
#
#   source scripts/env.sh && $MIDTRAIN_PYTHON -m midtrain.train_continue --help
#
# Paths differ between the workstation and the cluster, so no script hard-codes one; site.env.example lists them.
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${repository_root}/site.env" ]]; then
  echo "missing ${repository_root}/site.env: copy site.env.example and fill in the paths" >&2
  return 1 2>/dev/null || exit 1
fi
source "${repository_root}/site.env"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"        # training and evaluation never reach the network; downloads unset it
export TOKENIZERS_PARALLELISM=false                 # the tokenizer's own threads fight with the data workers
