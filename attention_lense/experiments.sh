#!/bin/bash 
ckpt_dir="/lus/grand/projects/SuperBERT/mansisak/attn_lens_ckpts/gpt2_small/"
qsub -v "ckpt=$ckpt_dir+ckpt_0, l_num=0, model_name=gpt2-small" -N attnlen_0 simple_submit.pbs 
qsub -v "ckpt=$ckpt_dir+ckpt_1, l_num=1, model_name=gpt2-small" -N attnlen_1 simple_submit.pbs 
qsub -v "ckpt=$ckpt_dir+ckpt_2, l_num=2, model_name=gpt2-small" -N attnlen_2 simple_submit.pbs 
qsub -v "ckpt=$ckpt_dir+ckpt_3, l_num=3, model_name=gpt2-small" -N attnlen_3 simple_submit.pbs 
qsub -v "ckpt=$ckpt_dir+ckpt_4, l_num=4, model_name=gpt2-small" -N attnlen_4 simple_submit.pbs 
qsub -v "ckpt=$ckpt_dir+ckpt_5, l_num=5, model_name=gpt2-small" -N attnlen_5 simple_submit.pbs 
qsub -v "ckpt=$ckpt_dir+ckpt_6, l_num=6, model_name=gpt2-small" -N attnlen_6 simple_submit.pbs
qsub -v "ckpt=$ckpt_dir+ckpt_7, l_num=7, model_name=gpt2-small" -N attnlen_7 simple_submit.pbs 
qsub -v "ckpt=$ckpt_dir+ckpt_8, l_num=8, model_name=gpt2-small" -N attnlen_8 simple_submit.pbs 
qsub -v "ckpt=$ckpt_dir+ckpt_9, l_num=9, model_name=gpt2-small" -N attnlen_9 simple_submit.pbs 
qsub -v "ckpt=$ckpt_dir+ckpt_10, l_num=10, model_name=gpt2-small" -N attnlen_10 simple_submit.pbs 
qsub -v "ckpt=$ckpt_dir+ckpt_11, l_num=11, model_name=gpt2-small" -N attnlen_11 simple_submit.pbs 
