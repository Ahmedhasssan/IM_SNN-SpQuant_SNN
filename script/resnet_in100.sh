export CUDA_VISIBLE_DEVICES=0,1


if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

model=resnet34
dataset="imagenet100"

lr=0.05
lamb=0.05
batch_size=32
optimizer="sgd"
lr_sch="cos"
T=2


log_file="training.log"
save_path="./save/${dataset}/QMem-3to1/w2a32/T${T}/${model}/${model}_lr${lr}_batch${batch_size}_${loss}loss/"
train_dir="/home2/jmeng15/data/imagenet-100/train/"
val_dir="/home2/jmeng15/data/imagenet-100/val/"

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 ./main.py \
    --model ${model} \
    --lr ${lr} \
    --lamb ${lamb} \
    --lr_sch ${lr_sch} \
    --optimizer ${optimizer} \
    --batch-size ${batch_size} \
    --epochs 100 \
    --dataset ${dataset} \
    --T ${T} \
    --log_file ${log_file} \
    --save_path ${save_path} \
    --train_dir ${train_dir} \
    --val_dir ${val_dir} \
    --lb -3.0 \
