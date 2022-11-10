#!/bin/sh

python benchmark_glue.py gpt2
python benchmark_glue.py bart
python benchmark_glue.py roberta



python benchmark_glue.py gpt2 ./models/gpt2-glue-tokenizer/ ./models/gpt2-glue_0.1/ 10
python benchmark_glue.py gpt2 ./models/gpt2-glue-tokenizer/ ./models/gpt2-glue_0.5/ 50
python benchmark_glue.py gpt2 ./models/gpt2-glue-tokenizer/ ./models/gpt2-glue_0.9/ 90

python benchmark_glue.py gpt2 ./models/gpt2-glue-tokenizer/ ./models/gpt2-glue_0.95/ 95
python benchmark_glue.py gpt2 ./models/gpt2-glue-tokenizer/ ./models/gpt2-glue_0.99/ 99
python benchmark_glue.py gpt2 ./models/gpt2-glue-tokenizer/ ./models/gpt2-glue/ 0
