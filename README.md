![medalpaca](https://user-images.githubusercontent.com/37253540/228315829-b22f793c-2dcd-4c03-a32d-43720085a7de.png)

# medAlpaca: a set of large language models, finetuned for medical question answering

## Project Overview
MedAlpaca expands upon both [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor) to offer an advanced suite of large language models specifically fine-tuned for medical question-answering and dialogue applications. Our primary objective is to deliver an array of open-source language models, paving the way for seamless development of medical chatbot solutions.

These models have been trained using a variety of medical texts, encompassing resources such as medical flashcards, wikis, and dialogue datasets. For more details on the data utilized, please consult the data section. 

| Model Name                | Description              | 
|---------------------------|--------------------------|
| medOPT-6.7B               | Description of the model |
| medOPT-13B                | Description of the model |
| medOPT-30B                | Description of the model |
| medAlpaca-6B              | Description of the model |
| medAlpaca-13B             | Description of the model |
| medAlpaca-30B             | Description of the model |

## Getting Started
Create a new virtual environment, e.g. with conda

```bash
conda create -n medalpaca python==3.9
```

Install the required packages:
```bash
pip install -r requirements.txt
```

Give the recency of LLaMA, HugginFace does not fully support it yet and you may run into [errors](https://github.com/tatsu-lab/stanford_alpaca#warning).
To mitigate these issues, we have specified the dependency versions that have proven effective for our purposes. Please refer to the pinned requirements for a smoother experience.

## Finetune The Models
All models have been trained on 8A100 GPUs with 80GB VRAM

### Finetune LLaMA 
If you have access to the [LLaMA](https://arxiv.org/abs/2302.13971) or [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) weights you can finetune the model with the following command. 
Just replace `<PATH_TO_LLAMA_WEIGHTS>` with the folder containing you LLaMA or Alpaca weights. 

```bash
torchrun --nproc_per_node=8 --master_port=<YOUR PORT> train.py \
   --model_name_or_path <PATH_TO_LLAMA_WEIGHTS> \
   --data_path medalpaca_small.json \
   --bf16 True \
   --output_dir models \
   --num_train_epochs 3 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 8 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True
```

### Finetune OPT
Finetune OPT ([Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068))

```bash
torchrun --nproc_per_node=8 --master_port=<YOUR PORT> train.py \
    --model_name_or_path "facebook/opt-6.7b" \
    --data_path medalpaca_small.json \
    --bf16 True \
    --output_dir models \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'OPTDecoderLayer' \
    --tf32 True
```


## Data
<img width="256" alt="Screenshot 2023-03-31 at 09 37 41" src="https://user-images.githubusercontent.com/37253540/229244284-72b00e82-0da1-4218-b08e-63864306631e.png">



The training data for this project was sourced from various resources. Firstly, we used Anki flashcards to automatically generate questions, from the front of the cards and anwers from the back of the card. Secondly, we generated medical question-answer pairs from [Wikidoc](https://www.wikidoc.org/index.php/Main_Page). We extracted paragraphs with relevant headings, and used Chat-GPT 3.5 to generate questions from the headings and using the corresponding paragraphs as answers. This dataset is still under development and we believe that approximately 70% of these question answer pairs are factual correct. Thirdly, we used StackExchange to extract question-answer pairs, taking the top-rated question from five categories: Academia, Bioinformatics, Biology, Fitness, and Health. Additionally, we used a dataset from https://arxiv.org/abs/2303.14070 consisting of 200,000 question-answer pairs, available at https://github.com/Kent0n-Li/ChatDoctor.

And the addition seven public biomedical datasets formatted in instruction tuning format is available to download here: https://drive.google.com/file/d/1YuHtEExQ4B_C4FPcHL3cAa0Y1Y2gCtuW/view?usp=share_link

| Source                      | n items |
|------------------------------|--------|
| ChatDoc large                | 200000 |
| wikidoc                      | 67704  |
| Stackexchange academia       | 40865  |
| Anki flashcards              | 33955  |
| Stackexchange biology        | 27887  |
| Stackexchange fitness        | 9833   |
| Stackexchange health         | 7721   |
| Wikidoc patient information  | 5942   |
| Stackexchange bioinformatics | 5407   |


We provide two datasets: `medalpaca_small` and `medalpaca_large`. 

`medalpaca_small` consists of 6000 Q/A pairs, consisting out of questions from the Wikidoc patient information and can be found in this repositories root directory
`medalpaca_large` consists of the whole dataset. You can apply here for access. Please note, that we are still in the process of cleaning and optimizing this dataset. 

## Benchmarks
<img width="256" alt="benchmarks" src="https://user-images.githubusercontent.com/37253540/229249302-20ff8a88-95b4-42a3-bdd8-96a9dce9a92b.png">

We benchmarked all models on the USMLE self assessment. 

| Model Name                | USMLE Step1              | USMLE Step1              | USMLE Step1              | 
|---------------------------|--------------------------|--------------------------|--------------------------|
| Vanillastanford Alpaca 7b |                          |                          |                          |
| medAlpaca 7b - No Lora    |                          |                          |                          |
| medAlpaca-7B              |                          |                          |                          |
| medAlpaca-13B             |                          |                          |                          |
| medAlpaca-30B             |                          |                          |                          |
| medAlpaca-65B             |                          |                          |                          |

## Chat with medAlpaca

TODO: Add Docker + WebApp
