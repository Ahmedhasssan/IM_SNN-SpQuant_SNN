export CUDA_VISIBLE_DEVICES=0


if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

model=vgg9
dataset="dvscifar10"
data_path="/home/ahasssan/QESNN/new_pruning/QESNN/data/dvs_cifar10_16"

lr=0.001
lamb=0.90
batch_size=24
optimizer="adam"
lr_sch="cos"
T=10
date="7_28"


log_file="training.log"
save_path="./save/new_pruning/Conv_skipping_with_1/4_inch/${dataset}/fire_wire_temporal_spatial_mask/eps_0.4/${date}/T${T}/${model}/lr${lr}_batch${batch_size}_${loss}loss_run2/"
#resume="/home/ahasssan/QESNN/new_pruning/QESNN/save/new_pruning/Conv_skipping_with_less_inch/dvscifar10/fire_wire_temporal_spatial_mask/eps_0.4/7_28/T10/vgg9/lr0.001_batch24_loss_run2/model_best.pth.tar"

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 ./main.py \
    --model ${model} \
    --lr ${lr} \
    --lamb ${lamb} \
    --lr_sch ${lr_sch} \
    --optimizer ${optimizer} \
    --batch-size ${batch_size} \
    --epochs 200 \
    --dataset ${dataset} \
    --T ${T} \
    --log_file ${log_file} \
    --save_path ${save_path} \
    --data_path ${data_path} \
    #--resume ${resume} \

