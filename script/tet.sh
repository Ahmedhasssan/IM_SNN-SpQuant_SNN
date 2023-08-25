model=resnet19
dataset="cifar10"

lr=0.01
batch_size=128
T=2

log_file="training.log"
save_path="./save/${dataset}/TETBaseline/T${T}/${model}/${model}_lr${lr}_batch${batch_size}/"

python ./main_ddp.py \
    --lr ${lr} \
    --lamb 0.05 \
    --batch-size ${batch_size} \
    --T ${T} \
    --lt False \
    --epochs 300 \
    --log_file ${log_file} \
    --save_path ${save_path} \