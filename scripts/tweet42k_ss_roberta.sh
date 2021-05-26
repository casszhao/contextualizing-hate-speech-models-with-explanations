#!/bin/bash

max_seeds=10 #run how many times
current_seed=0

while(( $current_seed < $max_seeds ))
do
    CUDA_LAUNCH_BLOCKING=1 python ./run_ss-roberta.py --do_train --do_lower_case --data_dir ./data/tweet42k --bert_model roberta-base --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --neutral_words_file ./data/identity.csv --num_train_epochs 20 --early_stop 5 --output_dir runs/tweet42k_roberta_seed_$current_seed --seed $current_seed --task_name ws --negative_weight 0.156
    let current_seed++
done






