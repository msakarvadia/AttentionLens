#!/bin/bash 

model_name="gpt2_small"

ckpt_dir="/lus/grand/projects/SuperBERT/mansisak/attn_lens_ckpts/${model_name}/ckpt_"
job_name="${model_name}_attnlen_"

for layer in {0..11}
do
    echo $layer
    qsub -v "ckpt=${ckpt_dir}${layer}, l_num=${layer}, model_name=$model_name" -N ${job_name}${layer} simple_submit.pbs 
done
