# source /opt/conda/bin/activate deyo
source /opt/conda/bin/activate nayer
# source /opt/conda/bin/activate foa
# 2 4 5 6 7
# export HOME=/mnt/cephfs/home/chendeyu
# 0 2 3 4 6 7
python3 datafree_kd.py \
    --data_root /chenyaofo/cgh/cowork/cdy/research/NAYER/data\
    --dataset cifar10 \
    --method nayer \
    --gpu 7 \
    --teacher resnet34 \
    --student resnet34 \
    --save_dir /chenyaofo/cgh/cowork/cdy/research/NAYER/outputs/ \
    --log_tag _800ep_5wp_40gsteps_2gloop_10gwploop_0.2lr_400kdsteps_64bs \
    --epochs 800 \
    --warmup 5 \
    --g_steps 40 \
    --g_loops 2 \
    --gwp_loops 10 \
    --lr 0.2 \
    --kd_steps 400 \
    --batch_size 64 \
    --synthesis_batch_size 64 \
    --print_freq 40 

    

# iclr-2026
# --data_root /chenyaofo/datasets/TTA/
    # --evaluate_only \
