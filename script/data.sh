
dataset="dvscifar10"
T=16
save_path="./dvs/pt/${dataset}T${T}/"
data_root="./dvs/${dataset}/"

python ./setup.py \
    --root_dir ${data_path} \
    --dataset ${dataset} \
    --T ${T} \
    --save_dir ${save_path} \

