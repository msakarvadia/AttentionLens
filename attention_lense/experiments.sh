#!/bin/bash 
qsub -v "ckpt=ckpt_0, l_num=0" -N attnlen_0 simple_submit.pbs 
qsub -v "ckpt=ckpt_1, l_num=1" -N attnlen_1 simple_submit.pbs 
qsub -v "ckpt=ckpt_2, l_num=2" -N attnlen_2 simple_submit.pbs 
qsub -v "ckpt=ckpt_3, l_num=3" -N attnlen_3 simple_submit.pbs 
qsub -v "ckpt=ckpt_4, l_num=4" -N attnlen_4 simple_submit.pbs 
qsub -v "ckpt=ckpt_5, l_num=5" -N attnlen_5 simple_submit.pbs 
qsub -v "ckpt=ckpt_6, l_num=6" -N attnlen_6 simple_submit.pbs
qsub -v "ckpt=ckpt_7, l_num=7" -N attnlen_7 simple_submit.pbs 
qsub -v "ckpt=ckpt_8, l_num=8" -N attnlen_8 simple_submit.pbs 
qsub -v "ckpt=ckpt_9, l_num=9" -N attnlen_9 simple_submit.pbs 
qsub -v "ckpt=ckpt_10, l_num=10" -N attnlen_10 simple_submit.pbs 
qsub -v "ckpt=ckpt_11, l_num=11" -N attnlen_11 simple_submit.pbs 
