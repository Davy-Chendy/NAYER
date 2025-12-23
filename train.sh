# source /opt/conda/bin/activate deyo
# source /opt/conda/bin/activate nayer
source /lichenghao/miniconda38/bin/activate nayer
# source /opt/conda/bin/activate foa
# 2 4 5 6 7
# export HOME=/mnt/cephfs/home/chendeyu
# 0 2 3 4 6 7
python3 train_scratch.py \
    --data_root /lichenghao/cdy/research/NAYER/data \
    --dataset cifar10 \
    --model resnet34 \
    --batch-size 256 \
    --lr 0.05 \
    --epoch 1000 \
    --gpu 7 \
    -p 40 

    

# iclr-2026
# --data_root /chenyaofo/datasets/TTA/
    # --evaluate_only \
# _120ep_20wp_40gsteps_10glife_2gloop_10gwploop_0.2lr_400kdsteps_512bs_400sbs \
# _800ep_20wp_40gsteps_10glife_2gloop_10gwploop_0.2lr_800kdsteps_20T_512bs_400sbs