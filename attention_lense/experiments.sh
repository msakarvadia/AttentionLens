#!/bin/bash 
qsub -v "ckpt=ckpt_0, l_num=0" simple_submit.pbs
qsub -v "ckpt=ckpt_1, l_num=1" simple_submit.pbs
qsub -v "ckpt=ckpt_2, l_num=2" simple_submit.pbs
qsub -v "ckpt=ckpt_3, l_num=3" simple_submit.pbs
qsub -v "ckpt=ckpt_4, l_num=4" simple_submit.pbs
qsub -v "ckpt=ckpt_5, l_num=5" simple_submit.pbs
qsub -v "ckpt=ckpt_6, l_num=6" simple_submit.pbs
qsub -v "ckpt=ckpt_7, l_num=7" simple_submit.pbs
qsub -v "ckpt=ckpt_8, l_num=8" simple_submit.pbs
qsub -v "ckpt=ckpt_9, l_num=9" simple_submit.pbs
qsub -v "ckpt=ckpt_10, l_num=10" simple_submit.pbs
qsub -v "ckpt=ckpt_11, l_num=11" simple_submit.pbs
