#!/bin/bash 

for model_name in gpt2
do
    echo $model_name
    ckpt_dir="/home/pettyjohnjn/TL_AttnLens/checkpoint/${model_name}/ckpt_"
    job_name="${model_name}_attnlen_"

    if [ $model_name == "gpt2" ];then 
        declare -i num_layers=12
    else
        declare -i num_layers=36
    fi

    for (( layer=0; layer<$num_layers; layer++ ))
    do
        echo $layer
        ./simple_submit.pbs  -v "l_num=${layer}" -N ${job_name}${layer}
    done

done
