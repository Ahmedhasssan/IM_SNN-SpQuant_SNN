export CUDA_VISIBLE_DEVICES=0


if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

model=vgg9
dataset="dvscifar10"
date="4_17"
data_path="/home/ahasssan/QESNN/new_pruning/QESNN/data/dvs_cifar10_16"

lr=0.001
lamb=0.90
batch_size=8
optimizer="adam"
lr_sch="cos"
T=10


log_file="training.log"
save_path="/home/ahasssan/QESNN/new_pruning/QESNN/save/Pruning/fire_wire_temporal/spatial_mask/6_24/dvscifar10/6_24/Baseline/w32a32/T10/vgg9/vgg9_lr0.001_batch4_loss_run2/eval/"
pretrained_model="/home/ahasssan/QESNN/new_pruning/QESNN/save/Pruning/fire_wire_temporal/spatial_mask/6_24/dvscifar10/6_24/Baseline/w32a32/T10/vgg9/vgg9_lr0.001_batch4_loss_run2/model_best.pth.tar"

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
    --fine_tune \
    --resume ${pretrained_model} \
    --evaluate \

