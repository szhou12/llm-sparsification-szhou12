import numpy as np
import datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import json
import sys

# key = model_name
# value = (tokenizer_name, list of saved models)
CHECKPOINTS = {
    'gpt2': ('gpt2', ["gpt2", "./models/gpt2_0.1/", "./models/gpt2_0.5/", "./models/gpt2_0.9/", "./models/gpt2_0.95/", "./models/gpt2_0.99/"]),
    'bart': ('facebook/bart-large', ["facebook/bart-large", "./models/bart_0.1/", "./models/bart_0.5/", "./models/bart_0.9/", "./models/bart_0.95/", "./models/bart_0.99/"]),
    'roberta': ('roberta-large', ["roberta-large", "./models/roberta_0.1/", "./models/roberta_0.5/", "./models/roberta_0.9/", "./models/roberta_0.95/", "./models/roberta_0.99/"]),
}

SPARSE_PERCENT = [0, 10, 50, 90, 95, 99]
context_length = 128


def save2Json(metrics, filename):
    json_object = json.dumps(metrics, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(json_object)



def run_benchmark(model_name):
    tokenizer_checkpoint, model_checkpoint_list = CHECKPOINTS[model_name]

    raw_datasets = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    ## Tokenization
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    def tokenize_function(samples):
        outputs = tokenizer(
            samples["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    ## Feed into Trainer
    for idx, model_checkpoint in enumerate(model_checkpoint_list):
        config = AutoConfig.from_pretrained(
            model_checkpoint,
            vocab_size=len(tokenizer),
            n_ctx=context_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = AutoModelForCausalLM.from_config(config)

        training_args = TrainingArguments(
            output_dir=f'output/glue-{model_name}_{SPARSE_PERCENT[idx]}/',
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            # num_train_epochs = 1,
        )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        trains = trainer.train()
        # predictions = trainer.predict(tokenized_datasets['validation'])
        save2Json(trains.metrics, filename=f'output/clm-{model_name}_{SPARSE_PERCENT[idx]}.json')


def run_benchmark_single(model_name, tokenizer_checkpoint, model_checkpoint, SPARSE_PERCENT):
    raw_datasets = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    ## Tokenization
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    def tokenize_function(samples):
        outputs = tokenizer(
            samples["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    ## Feed into Trainer
    config = AutoConfig.from_pretrained(
        model_checkpoint,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = AutoModelForCausalLM.from_config(config)

    training_args = TrainingArguments(
        output_dir=f'output/clm-{model_name}_{SPARSE_PERCENT}/',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        # num_train_epochs = 1,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trains = trainer.train()
    # predictions = trainer.predict(tokenized_datasets['validation'])
    save2Json(trains.metrics, filename=f'output/clm-{model_name}_{SPARSE_PERCENT}.json')


if __name__ == "__main__":
    model_name = sys.argv[1].strip()

    # tokenizer_checkpoint = sys.argv[2].strip()
    # model_checkpoint = sys.argv[3].strip()
    # SPARSE_PERCENT = sys.argv[4].strip()

    # run_benchmark_single(
    #     model_name, 
    #     tokenizer_checkpoint, 
    #     model_checkpoint, 
    #     SPARSE_PERCENT
    # )


    run_benchmark(model_name)
