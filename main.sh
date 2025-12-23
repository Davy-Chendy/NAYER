# source /opt/conda/bin/activate deyo
# source /opt/conda/bin/activate nayer
source /lichenghao/miniconda38/bin/activate nayer
# source /opt/conda/bin/activate foa
# 2 4 5 6 7
# export HOME=/mnt/cephfs/home/chendeyu
# 0 2 3 4 6 7
python3 datafree_kd.py \
    --data_root /lichenghao/cdy/research/NAYER/data \
    --dataset cifar10 \
    --method nayer \
    --gpu 5 \
    --teacher resnet34 \
    --student gn-resnet34 \
    --save_dir /root/cdy_outputs/nayer_outputs/ \
    --log_tag _800ep20wp_30gsteps_10glife_2gloop_10gwploop_0.005lr_400kdsteps_20T_256bs_200sbs_only_norm_adapt \
    --epochs 800 \
    --warmup 20 \
    --g_steps 30 \
    --g_life 10 \
    --g_loops 2 \
    --gwp_loops 10 \
    --lr 0.005 \
    --kd_steps 400 \
    --T 20 \
    --batch_size 256 \
    --synthesis_batch_size 200 \
    --print_freq 40 \

    

# iclr-2026
# --data_root /chenyaofo/datasets/TTA/
    # --evaluate_only \
# _120ep_20wp_40gsteps_10glife_2gloop_10gwploop_0.2lr_400kdsteps_512bs_400sbs \