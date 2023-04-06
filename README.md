![medalpaca](https://user-images.githubusercontent.com/37253540/228315829-b22f793c-2dcd-4c03-a32d-43720085a7de.png)

# medAlpaca: Finetuned Large Language Models for Medical Question Answering

## Project Overview
MedAlpaca expands upon both [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and 
[AlpacaLoRA](https://github.com/tloen/alpaca-lora) to offer an advanced suite of large language 
models specifically fine-tuned for medical question-answering and dialogue applications. 
Our primary objective is to deliver an array of open-source language models, paving the way for 
seamless development of medical chatbot solutions.

These models have been trained using a variety of medical texts, encompassing resources such as 
medical flashcards, wikis, and dialogue datasets. For more details on the data utilized, please consult the data section. 

## Getting Started
Create a new virtual environment, e.g. with conda

```bash
conda create -n medalpaca python>=3.9
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Training of medAlpaca
<img width="256" alt="training your alpaca" src="https://user-images.githubusercontent.com/37253540/229250535-98f28e1c-0a8e-46e7-9e61-aeb98ef115cc.png">

### Memory Requirements
We have benchmarked the needed GPU memory as well as the approximate duration per epoch 
for finetuning LLaMA 7b on the Medical Meadow small dataset (~6000 Q/A pairs) on a single GPU:


| Model    | 8bit trainig | LoRA  | fp16  | bf16  | VRAM Used | Gradient cktp | Duration/epoch |
|----------|--------------|-------|-------|-------|-----------|---------------|----------------|
| LLaMA 7b | True         | True  | True  | False | 8.9 GB    | False         | 77:30          |
| LLaMA 7b | False        | True  | True  | False | 18.8 GB   | False         | 14:30          |
| LLaMA 7b | False        | False | True  | False | OOM       | False         | -              | 
| LLaMA 7b | False        | False | False | True  | 79.5 GB   | True          | 35:30          |
| LLaMA 7b | False        | False | False | False | OOM       | True          | -              |

### Train medAlpaca based on LLaMA 
If you have access to the [LLaMA](https://arxiv.org/abs/2302.13971) or [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) 
weights you can finetune the model with the following command. 
Just replace `<PATH_TO_LLAMA_WEIGHTS>` with the folder containing you LLaMA or Alpaca weights. 

```bash
python medalpaca/train.py \
    --model PATH_TO_LLAMA_WEIGHTS \
    --data_path medical_meadow_small.json \
    --output_dir 'output' \
    --train_in_8bit True \  
    --use_lora True \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --global_batch_size 128 \
    --per_device_batch_size 8 \
```
Per default the script performs mixed precision training.  
You can toggle 8bit training with the `train_in_8bit` flag. 
While 8 bit training currently only works with `use_lora True`, however you can use
LoRA without 8 bit training. 
It is also able to train other models such as `facebook/opt-6.7` with the above script. 

## Data
<img width="256" alt="Screenshot 2023-03-31 at 09 37 41" src="https://user-images.githubusercontent.com/37253540/229244284-72b00e82-0da1-4218-b08e-63864306631e.png">

To ensure your cherished llamas and alpacas are well-fed and thriving, 
we have diligently gathered high-quality biomedical open-source datasets 
and transformed them into instruction tuning formats. 
We have dubbed this endeavor **Medical Meadow**. 
Medical Meadow currently encompasses roughly 1.5 million data points across a diverse range of tasks, 
including openly curated medical data transformed into Q/A pairs with OpenAI's `gpt-3.5-turbo`
and a collection of established NLP tasks in the medical domain. 
Please note, that not all data is of the same quantitiy and quality and you may need tp subsample 
the data for training your own model. 
We will persistently update and refine the dataset, and we welcome everyone to contribute more 'grass' to Medical Meadow!

### Data Overview

| Name                 |  Source                                                                 |  n       |  n included in training |
|----------------------|-------------------------------------------------------------------------|----------|-------------------------|
| Medical Flashcards   |  [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)  |  33955  |  33955                 |
| Wikidoc              |  [medalpaca/medalpaca/medical_meadow_wikidoc](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc)    |  67704  |  10000                 |
| Wikidoc Patient Information | [medalpaca/medalpaca/medical_meadow_wikidoc_patient_information](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information)    |  5942 |  5942 |
| Stackexchange academia |  [medalpaca/medalpaca/medical_meadow_stack_exchange](https://huggingface.co/medalpaca/datasets/medalpaca/medical_meadow_stackexchange)    |  40865  |  40865                 |
| Stackexchange biology |  [medalpaca/medalpaca/medical_meadow_stack_exchange](https://huggingface.co/medalpaca/datasets/medalpaca/medical_meadow_stackexchange)    |  27887  |  27887                 |
| Stackexchange fitness |  [medalpaca/medalpaca/medical_meadow_stack_exchange](https://huggingface.co/medalpaca/datasets/medalpaca/medical_meadow_stackexchange)    |  9833  | 9833                 |
| Stackexchange health |  [medalpaca/medalpaca/medical_meadow_stack_exchange](https://huggingface.co/medalpaca/datasets/medalpaca/medical_meadow_stackexchange)    |  7721  |  7721                 |
| Stackexchange bioinformatics |  [medalpaca/medalpaca/medical_meadow_stack_exchange](https://huggingface.co/datasets/medalpaca/medical_meadow_stackexchange)    |  5407  |  5407                |
| USMLE Self Assessment Step 1 |  [medalpaca/medalpaca/medical_meadow_usmle_self](https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self)    |  119  |  92 (test only)              |
| USMLE Self Assessment Step 2 |  [medalpaca/medalpaca/medical_meadow_usmle_self](https://huggingface.co/vmedalpaca/medical_meadow_usmle_self)    |  120  |  110  (test only)              |
| USMLE Self Assessment Step 3 |  [medalpaca/medalpaca/medical_meadow_usmle_self](vhuggingface.co/datasets/medalpaca/medical_meadow_usmle_self)    |  135  |  122  (test only)             |
| MEDIQA               | [original](https://osf.io/fyg46/?view_only=), [preprocessed](https://huggingface.co/datasets/medalpaca/medical_meadow_mediqa) |  2208    |  2208 |
| CORD-19              | [original](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge ), [preprocessed](https://huggingface.co/datasets/medalpaca/medical_meadow_cord19) |  1056660    |  50000 |
| MMMLU               | [original](https://github.com/hendrycks/test), [preprocessed](https://huggingface.co/datasets/medalpaca/medical_meadow_mmmlu) |  3787    |  3787 |
| Pubmed Health Advice | [original](https://aclanthology.org/D19-1473/), [preprocessed](vhuggingface.co/datasets/medalpaca/health_advice) |  10178    |  10178 |
| Pubmed Causal               | [original](https://aclanthology.org/2020.coling-main.427/    ), [preprocessed](https://huggingface.co/datasets/medalpaca/medical_meadow_pubmed_causal) |  2446    |  2446 |
| ChatDoctor               | [original](https://github.com/Kent0n-Li/ChatDoctor  ) |  215000    |  10000 |
| OpenAssistant | [original](https://huggingface.co/OpenAssistant) |  9209   | 9209     |


### Data description
please refer to [DATA_DESCRIPTION.md](DATA_DESCRIPTION.md)


## Benchmarks
<img width="256" alt="benchmarks" src="https://user-images.githubusercontent.com/37253540/229249302-20ff8a88-95b4-42a3-bdd8-96a9dce9a92b.png">

We are benchmarking all models on the USMLE self assessment, which is available at this [link](https://www.usmle.org/prepare-your-exam).
Note, that we removed all questions with images, as our models are not multimodal. 

| **Model**                                | **Step1**      | **Step2**      | **Step3**      |
|------------------------------------------|----------------|----------------|----------------|
| [LLaMA 7b](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) | 0.174          | 0.109          | nan            |
| [Alpaca 7b naive](https://github.com/tatsu-lab/stanford_alpaca)  | 0.243          | 0.222          | 0.329          |
| [Alpaca 7b LoRA](https://github.com/tloen/alpaca-lora) | 0.261          | 0.264          | 0.266          |
| [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor) | 0.187          | 0.185          | 0.148          |
| MedAlpaca 7b                             | 0.261          | 0.300          | **0.363**      |
| MedAlpaca 7b LoRA 8bit                   | 0.196          | 0.209          | 0.185          |
| MedAlpaca 13b LoRA 8bit                  | 0.217          | 0.155          | 0.234          |
| MedAlpaca 30b LoRA 8bit                  | **0.315**      | **0.327**      | 0.355          |

We are continuously working on improving the training as well as our evaluation prompts. 
Expect this table to change quite a bit. 


## Access the models
Visit the zoo and have a look at our alpacas here: https://huggingface.co/medalpaca

It should be obvious, but the models provided on this platform are shared for research purposes 
only and should not be used in any healthcare applications or settings. 
While we are excited to showcase our experimental models, please be aware that they have not undergone 
extensive testing or validation, and their reliability cannot be guaranteed. 
We kindly ask you to exercise caution when using these models, 
and we appreciate your understanding as we continue to explore and develop this innovative technology.


## Chat with medAlpaca
<img width="256" alt="chat-lama" src="https://user-images.githubusercontent.com/37253540/229261366-5cce9a60-176a-471b-80fd-ba390539da72.png">

A Convenient interface to our models is coming soon
