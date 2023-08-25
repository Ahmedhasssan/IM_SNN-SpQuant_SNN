export CUDA_VISIBLE_DEVICES=0


if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

model=resnet19
dataset="cifar10"

lr=0.01
lamb=0.05
batch_size=128
optimizer="adam"
lr_sch="cos"
T=2


log_file="training.log"
save_path="./save/${dataset}/pruning/spatial_temporal/T${T}/${model}/${model}_lr${lr}_batch${batch_size}_${loss}loss/"

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 ./main.py \
    --model ${model} \
    --lr ${lr} \
    --lamb ${lamb} \
    --lr_sch ${lr_sch} \
    --optimizer ${optimizer} \
    --batch-size ${batch_size} \
    --epochs 300 \
    --dataset cifar10 \
    --T ${T} \
    --log_file ${log_file} \
    --save_path ${save_path} \
