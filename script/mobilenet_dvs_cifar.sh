export CUDA_VISIBLE_DEVICES=0


if [ ! -d "$DIRECTORY" ]; then
    mkdir ../save
fi

model=mobilenet_tiny
dataset="dvscifar10"
date="4_17"
data_path="/home/jmeng15/QESNN/dvs/dvs_cifar10_16steps/"

lr=1e-3
lamb=0.45
batch_size=32
optimizer="adam"
lr_sch="cos"
T=16
wbit=32


log_file="training.log"
save_path="./save/${dataset}/${date}/Baseline/w${wbit}a32/T${T}/${model}/${model}_lr${lr}_batch${batch_size}_${loss}loss/"

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
    --wbit ${wbit} \

