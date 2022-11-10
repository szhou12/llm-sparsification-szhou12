import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
import datasets


raw_datasets = datasets.load_dataset('glue', 'mrpc')

# key=tokenizer_name, value=list of models
checkpoint_map = {
    'facebook/bart-large': ["./models/bart_0.1/", "./models/bart_0.5/", "./models/bart_0.9/", "./models/bart_0.95/", "./models/bart_0.99/", "facebook/bart-large"],
    'roberta-large': ["./models/roberta_0.1/", "./models/roberta_0.5/", "./models/roberta_0.9/", "./models/roberta_0.95/", "./models/roberta_0.99/", "roberta-large"],
}
tokenizer_checkpoint_list = ['roberta-large']

checkpoint = 'roberta-large'
# checkpoint = 'facebook/bart-large'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)