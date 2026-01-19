# source /opt/conda/bin/activate deyo
# source /opt/conda/bin/activate nayer
# source /lichenghao/miniconda38/bin/activate nayer
source /home/troynsc/miniconda3/bin/activate nayer
# source /opt/conda/bin/activate foa
# 2 4 5 6 7
# export HOME=/mnt/cephfs/home/chendeyu
# 0 2 3 4 6 7
python3 datafree_kd.py \
    --data_root /home/troynsc/cgh/cdy/research/NAYER/data \
    --dataset cifar10 \
    --method nayer \
    --gpu 0 \
    --teacher resnet34 \
    --student gn-resnet34 \
    --save_dir /home/troynsc/cgh/cdy/research/NAYER/outputs/ \
    --log_tag 100ep_0wp_40gsteps_10glife_2gloop_10gwploop_0.1lr_400kdsteps_512bs_400sbs_NotLoadBN_TrainingALL \
    --epochs 100 \
    --warmup 0 \
    --g_steps 40 \
    --g_life 10 \
    --g_loops 2 \
    --gwp_loops 10 \
    --lr 0.1 \
    --kd_steps 400 \
    --T 20 \
    --batch_size 512 \
    --synthesis_batch_size 400 \
    --print_freq 40 \
    --gn_replaced \
    --training_params all \
    # --bn_pretrained \

    
# --bn_pretrained \
# iclr-2026
# --data_root /chenyaofo/datasets/TTA/
    # --evaluate_only \
# _120ep_20wp_40gsteps_10glife_2gloop_10gwploop_0.2lr_400kdsteps_512bs_400sbs \
#  _800ep20wp_30gsteps_10glife_2gloop_10gwploop_0.005lr_400kdsteps_20T_256bs_200sbs_only_norm_adapt