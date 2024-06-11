#!/bin/bash 

for model_name in gpt2
do
    echo $model_name
    ckpt_dir="/lus/grand/projects/SuperBERT/mansisak/attn_lens_ckpts/${model_name}/ckpt_"
    job_name="${model_name}_attnlen_"

    if [ $model_name == "gpt2" ];then 
        declare -i num_layers=12
    else
        declare -i num_layers=36
    fi

    for (( layer=0; layer<$num_layers; layer++ ))
    do
        echo $layer
        parallel -v "ckpt=${ckpt_dir}${layer}, l_num=${layer}, model_name=$model_name" -N ${job_name}${layer} simple_submit.pbs 
    done

done
