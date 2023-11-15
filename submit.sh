#!/usr/bin/env sh


set -eu  # o pipefail

GPU=${GPU:-0,1}
N_GPUS=${N_GPUS:-1}
PORT=${PORT:-29500}

#data_dir=$SLURM_TMPDIR/data
#chkps_dir=~/scratch/neuroswipe
data_dir=./data
chkps_dir=./chkps

backbone=lstm_bi
#backbone=transformer

N_EPOCHS=400
T_MAX=400

n_folds=10

BS=512
fold=0
hs=2048
ed=128
nl=1

CHECKPOINT="${chkps_dir}"/"${backbone}"_f"${fold}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_h"${hs}"_e"${ed}"_nl"${nl}"_lr1e3_2_cws_nac_eos_mask_aug_FULLTRAIN

#MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
CUDA_VISIBLE_DEVICES="${GPU}" python \
    ./src/submit.py \
        --train-df "${data_dir}/test.csv" \
        --grid-path "${data_dir}/grid.json" \
        --voc-path "${data_dir}/voc.txt" \
        --backbone "${backbone}" \
        --hidden-size "${hs}" \
        --emb-dim "${ed}" \
        --n-layers "${nl}" \
        --fold "${fold}" \
        --n-fold "${n_folds}" \
        --batch-size "${BS}" \
        --load $CHECKPOINT/model_best.pth \

