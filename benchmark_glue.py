import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
import datasets
from datasets import load_metric
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import sys
import json


# key = model_name
# value = (tokenizer_name, list of saved models)
CHECKPOINTS = {
    'gpt2': ('./models/gpt2-glue-tokenizer/', ["./models/gpt2-glue/", "./models/gpt2-glue_0.1/", "./models/gpt2-glue_0.5/", "./models/gpt2-glue_0.9/", "./models/gpt2-glue_0.95/", "./models/gpt2-glue_0.99/"]),
    'bart': ('facebook/bart-large', ["facebook/bart-large", "./models/bart_0.1/", "./models/bart_0.5/", "./models/bart_0.9/", "./models/bart_0.95/", "./models/bart_0.99/"]),
    'roberta': ('roberta-large', ["roberta-large", "./models/roberta_0.1/", "./models/roberta_0.5/", "./models/roberta_0.9/", "./models/roberta_0.95/", "./models/roberta_0.99/"]),
}

SPARSE_PERCENT = [0, 10, 50, 90, 95, 99]


def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def save2Json(metrics, filename):
    json_object = json.dumps(metrics, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(json_object)


def run_benchmark(model_name):
    tokenizer_checkpoint, model_checkpoint_list = CHECKPOINTS[model_name]

    raw_datasets = datasets.load_dataset('glue', 'mrpc')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    def tokenize_function(sample):
        return tokenizer(sample['sentence1'], sample['sentence2'], truncation=True)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for idx, model_checkpoint in enumerate(model_checkpoint_list):
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

        training_args = TrainingArguments(
            # num_train_epochs = 1,
            output_dir=f'output/glue-{model_name}_{SPARSE_PERCENT[idx]}/',
        )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator, 
            tokenizer=tokenizer,
            # compute_metrics=compute_metrics
        )
        print(f"Start training {model_checkpoint}...")

        trains = trainer.train()
        # predictions = trainer.predict(tokenized_datasets['validation'])
        save2Json(trains.metrics, filename=f'output/glue-{model_name}_{SPARSE_PERCENT[idx]}.json')


def run_benchmark_single(model_name, tokenizer_checkpoint, model_checkpoint, SPARSE_PERCENT):
    raw_datasets = datasets.load_dataset('glue', 'mrpc')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    def tokenize_function(sample):
        return tokenizer(sample['sentence1'], sample['sentence2'], truncation=True)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    training_args = TrainingArguments(
        num_train_epochs = 1,
        output_dir=f'output/glue-{model_name}_{SPARSE_PERCENT}/',
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator, 
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )
    print(f"Start training {model_checkpoint}...")

    trains = trainer.train()
    # predictions = trainer.predict(tokenized_datasets['validation'])
    save2Json(trains.metrics, filename=f'output/glue-{model_name}_{SPARSE_PERCENT}.json')


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




