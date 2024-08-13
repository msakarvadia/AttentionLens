#!/bin/bash 

for model_name in gpt2
do
    echo $model_name

    job_name="${model_name}_attnlen_"

    if [ $model_name == "gpt2" ];then 
        declare -i num_layers=12
    else
        declare -i num_layers=36
    fi

    for (( layer=0; layer<8; layer++ ))
    do
        echo $layer
        ckpt_dir="/home/pettyjohnjn/AttentionLens/checkpoint3/${model_name}/ckpt_"
        qsub -v "ckpt=${ckpt_dir}${layer}, l_num=${layer}, model_name=$model_name" -N ${job_name}${layer} simple_submit.pbs 
    done

done