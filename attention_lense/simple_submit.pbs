#!/bin/bash 
#PBS -l select=10
#PBS -l walltime=12:00:00
#PBS -q ATPESC_BigRun
#PBS -l filesystems=home:grand:eagle
#PBS -A ATPESC2023

cd "/lus/grand/projects/SuperBERT/mansisak/AttentionLens/attention_lense"
echo "working dir: "
pwd

# USAGE:
#   
#   To launch pretraining with this script, first provide the training
#   config, training data path, and output directory via the CONFIG,
#   DATA, and OUTPUT_DIR variables at the top of the script.
#
#   Run locally on a compute node:
#
#     $ ./run_pretraining.cobalt 
#
#   Submit as a Cobalt job (edit Cobalt arguments in the #COBALT directives
#   at the top of the script):
#
#     $ qsub-gpu path/to/run_pretraining.cobalt
#
#   Notes: 
#     - training configuration (e.g., # nodes, # gpus / node, etc.) will be
#       automatically inferred
#     - additional arguments to run_pretraining.py can be specified by
#       including them after run_pretraining.cobalt. E.g.,
#
#       $ ./run_pretraining.cobalt --steps 1000 --learning_rate 5e-4
#


# Figure out training environment
if [[ -z "${PBS_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MASTER_RANK=$(head -n 1 $PBS_NODEFILE)
    echo "Master Rank: "+$MASTER_RANK
    RANKS=$(tr '\n' ' ' < $PBS_NODEFILE)
    NNODES=$(< $PBS_NODEFILE wc -l)
fi


# Commands to run prior to the Python script for setting up the environment
PRELOAD="source /etc/profile ; "
PRELOAD+="module load conda ; "
PRELOAD+="conda activate ; "
PRELOAD+="conda activate /home/mansisak/.conda/envs/attnlens ; "
# PRELOAD+="module load conda/pytorch ; "
# PRELOAD+="conda activate /lus/theta-fs0/projects/SuperBERT/jgpaul/envs/pytorch-1.9.1-cu11.3 ; "
#PRELOAD+="module load conda ; "
#PRELOAD+="conda activate ; "
PRELOAD+="export OMP_NUM_THREADS=8 ; "

#PRELOAD="source /lus/grand/projects/SuperBERT/mansisak/AttentionLens/venv/bin/activate ; "
PRELOAD+="source ~/.bashrc ; "
PRELOAD+="export OMP_NUM_THREADS=8 ; "

# time python process to ensure timely job exit
TIMER="timeout 718m "

# torchrun launch configuration
LAUNCHER="python -m torch.distributed.run "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=auto --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--master_port=1222 --master_addr=$MASTER_RANK "
    #LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$MASTER_RANK "
fi

# Training script and parameters
#CMD="run_pretraining.py --input_dir $DATA --output_dir $OUTPUT_DIR --config_file $CONFIG " #Eventually we will use this type of command
#CMD="mlm_distributed.py"
echo "ckpt dir: "$ckpt
echo "layer num: "$l_num
echo "model name: "$model_name
CMD="train/train_pl.py --num_nodes $NNODES --model_name $model_name  --checkpoint_dir $ckpt --layer_number $l_num"

RANK=0
for NODE in $RANKS; do
    NODERANK=" --node_rank=$RANK "
    FULL_CMD=" $PRELOAD $LAUNCHER $NODERANK $CMD $@ "
    echo "Training Command: $FULL_CMD"

    if [[ "$NODE" == "$HOSTNAME" ]]; then

        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else

        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait

