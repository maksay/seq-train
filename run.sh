#!/bin/bash

# CMD arguments
mode=$1;
cam=$2;

# General settings
dp_name="DukeDataProvider";
dp_freq=0.33;
dp_size=6;
dt_size=5;

if [ "$cam" == "1" ]
then
    start_train=44157;
    start_fast=209000;
    start_eval=211000;
    start_test=221000;
    finish_test=352000;
fi

if [ "$cam" == "2" ]
    then
    start_train=46093;
    start_fast=211000;
    start_eval=213000;
    start_test=223000;
    finish_test=354000;
fi

if [ "$cam" == "3" ]
    then
    start_train=22456;
    start_fast=187000;
    start_eval=189000;
    start_test=199000;
    finish_test=330000;
fi

if [ "$cam" == "4" ]
    then
    start_train=18518;
    start_fast=183000;
    start_eval=185000;
    start_test=195000;
    finish_test=326000;
fi

if [ "$cam" == "5" ]
    then
    start_train=49700;
    start_fast=215000;
    start_eval=217000;
    start_test=227000;
    finish_test=357000;
fi

if [ "$cam" == "6" ]
    then
    start_train=27298;
    start_fast=192000;
    start_eval=194000;
    start_test=204000;
    finish_test=335000;
fi

if [ "$cam" == "7" ]
    then
    start_train=30732;
    start_fast=196000;
    start_eval=198000;
    start_test=208000;
    finish_test=339000;
fi

if [ "$cam" == "8" ]
    then
    start_train=2934;
    start_fast=168000;
    start_eval=170000;
    start_test=180000;
    finish_test=311000;
fi

# Frames used for training
train_frames="(${cam},${start_train},${start_fast})"

# Frames used for generating data - same frames, but divided between runners executed in parallel.
gendata_step=11800;

train_gendata_groups=();
for gendata_start in $(seq ${start_train} ${gendata_step} ${start_fast}); do
    gendata_end=$((${gendata_start} + ${gendata_step}));
    train_gendata_groups+=("(${cam},${gendata_start},${gendata_end})");
done

# Frames used for evaluation
fast_step=200;
fast_valid_groups=();
for fast_start in $(seq ${start_fast} ${fast_step} ${start_eval}); do
    fast_end=$((${fast_start} + ${fast_step}))
    fast_valid_groups+=("(${cam},${fast_start},${fast_end})");
done

# Execution-specific configs.

# Set of features
# bdif - difference between bboxes
# bbox - bounding box coordinates
# conf - bounding box confidence
# soc3 - social vicinity - 3 nearest bboxes (soc1|soc3|soc5)
# intp - flag of whether we interpolate a detection or have a real one
# appr - appearance
# dens - crowd density feature
export label_config=(\
 --label_config.features bdif-bbox-conf-soc3-intp\
 --label_config.score_fun idf_2\
 --label_config.output_fun iou_list\
 --label_config.app_dir "external/open-reid/"\
);

# Generally shouldn't be modified.
export train_data_provider_config=(\
 --train_data_provider_config.name ${dp_name}\
 --train_data_provider_config.frames ${train_frames}\
 --train_data_provider_config.frequency ${dp_freq}\
 --train_data_provider_config.scene_size ${dp_size}\
);
# Generally shouldn't be modified.
export train_input_data_provider_config=(\
 --input_data_provider_config.name ${dp_name}\
 --input_data_provider_config.frames "INVALID_DEFAULT"\
 --input_data_provider_config.frequency ${dp_freq}\
 --input_data_provider_config.scene_size ${dp_size}\
);
# Generally shouldn't be modified.
export valid_eval_data_provider_config=(\
 --input_data_provider_config.name ${dp_name}\
 --input_data_provider_config.frames "INVALID_DEFAULT"\
 --input_data_provider_config.frequency ${dp_freq}\
 --input_data_provider_config.scene_size ${dp_size}\
);

 # Model parameters
 #
 # Size of embedding layer and RNN cell
 # Batch size, regularization, learning rate
 # Number of RNN layers
 # Whether to predict ious, existence of gt, bbox shift to GT

export model_config=(\
 --model_config.name BidiRNNIoUPredictorModel\
 --model_config.embedding_size 300\
 --model_config.rnn_cell_size 300\
 --model_config.dropout 0.0\
 --model_config.l2_norm 0.0\
 --model_config.batch_size 100\
 --model_config.lr 0.001\
 --model_config.samples_per_epoch 1000000000\
 --model_config.predict_ious 1\
 --model_config.predict_labels 1\
 --model_config.predict_bboxes 1\
 --model_config.layers 1\
);

# MHT configuration.
#
export nms_config=(\
 --nms_config.name SimpleScoreAndNMSHypotheses\
 --nms_config.pairwise_min_iou 0.000000001\
 --nms_config.pairwise_max_dt ${dt_size}\
 --nms_config.nms_option start-0.9-ignore\
);

# Final greedy solution config
# Minimum score of hypothesis to go to the final solution
# Two hypotheses could only go to the final solution if iou less than predefined value
export final_solution_config=(\
 --final_solution_config.name GreedyFinalSolution\
 --final_solution_config.score_cutoff 0.6\
 --final_solution_config.iou_cutoff 0.65\
 --final_solution_config.no_overlap_policy 0\
);

# Name of the experiment.
# All experiment-related data will be in runs/experiment_name
experiment_name="duke_$cam";

export common_config=(\
 ${label_config[@]}\
 ${train_data_provider_config[@]}\
 ${model_config[@]}\
 ${nms_config[@]}\
 --experiment_name ${experiment_name}\
 --logging file);

# Generating the dataset
# Starts one master that trains the model
# and multiple slaves picking last checkpoints,
# running with it, and generating new data.
if [ "$1" == "gen_dataset" ]
    then
    CUDA_VISIBLE_DEVICES='' python main.py\
     ${common_config[@]}\
     --mode gen_dataset\
     & sleep 15;

    for train_group in ${train_gendata_groups[@]}; do
      CUDA_VISIBLE_DEVICES='' python main.py\
       ${common_config[@]}\
       ${train_input_data_provider_config[@]}\
       --input_data_provider_config.frames ${train_group}\
       --mode gen_data & sleep 5;
    done
fi


# Training
if [ "$1" == "train" ]
    then
    CUDA_VISIBLE_DEVICES='' python main.py\
     ${common_config[@]}\
     --mode train & sleep 15;
fi

# Evaluation. For training need to be run in parallel with train.
# Multiple slaves in parallel load checkpoints
# of the model to evaluate the performance. One master gathers
# their resutls, saves tensorboard summaries, and updates the
# best model.
if [ "$1" == "eval" ]
    then
    for fast_valid_group in ${fast_valid_groups[@]:0:5}; do
        CUDA_VISIBLE_DEVICES='' python main.py\
        ${common_config[@]}\
        ${valid_eval_data_provider_config[@]}\
        ${final_solution_config[@]}\
        --mode "continuous_eval_runner"\
        --input_data_provider_config.frames ${fast_valid_group}\
        & sleep 1;
    done
    #--final_solution_config.len_cutoff 5\

    CUDA_VISIBLE_DEVICES='' python main.py\
    ${common_config[@]}\
    --mode "continuous_eval_joiner"\
    --eval_runner_cnt 5\
    & sleep 1;
    #--eval_runner_cnt ${#fast_valid_groups[@]}\
fi

# Inference on all test sequence
test_start="${start_test}";
test_end="${finish_test}";
test_frames="($cam,${test_start},${test_end})";

# Inference is run with sequences of length 12,
# although training was performed with sequences of length 6.
if [ "$1" == "infer" ]
    then
    CUDA_VISIBLE_DEVICES='' python main.py\
     ${common_config[@]}\
     ${valid_eval_data_provider_config[@]}\
     ${final_solution_config[@]}\
     --mode "infer"\
     --input_data_provider_config.frames ${test_frames}\
     --input_data_provider_config.scene_size 12\
    & sleep 1;
fi
