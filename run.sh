#!/bin/sh

# python -m torch.distributed.launch --nproc_per_node 4 pytorch/language-modeling/run_clm.py \
#     --model_name_or_path "$(pwd)"/models/gpt2_0.1 \
#     --tokenizer_name gpt2 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --output_dir output


torchrun pytorch/language-modeling/run_clm.py \
    --model_name_or_path "$(pwd)"/models/gpt2_0.1 \
    --tokenizer_name gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --output_dir output

torchrun pytorch/language-modeling/run_clm.py --model_name_or_path "$(pwd)"/models/gpt2_0.1 --tokenizer_name gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --num_train_epochs 1 --output_dir output

torchrun pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 --tokenizer_name gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --num_train_epochs 1 --output_dir output


python -m torch.distributed.launch --nproc_per_node 4 pytorch/language-modeling/run_clm.py --model_name_or_path "$(pwd)"/models/gpt2_0.1 --tokenizer_name gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --output_dir output


torchrun pytorch/language-modeling/run_clm.py --model_name_or_path facebook/bart-large --tokenizer_name facebook/bart-large --output_dir output --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir

python -m torch.distributed.launch --nproc_per_node 4 pytorch/language-modeling/run_clm.py --model_name_or_path roberta-large --tokenizer_name "roberta-large" --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --output_dir test-clm-roberta-0.0-train/