# Medical Meadow

## Overview
*Medical Meadow* is a collection of medical datasets, some were explicitly created by us for this project, some are established medical datasets. If you use the *Medicl Meadow* for your research, please cite all datasets accordingly.
Currently the following datasets are part of our meadow.

### Self Created Datasets
These datasets were created by us for this project. Please cite our paper if you should use them. 

```Citation
Citation TBA
```

| Dataset                       | n       |
|-------------------------------|---------|
| Anki Flashcards               | 33,955  |
| Stack Exchange Academia       | 39,633  |
| Stack Exchange Biology        | 7,482   |
| Stack Exchange Fitness        | 3,026   |
| Stack Exchange Health         | 1,428   |
| Stack Exchange Bioinformatics | 906     |
| Wikidoc Living Textbook       | 67,704  |
| Wikidoc Patient Information   | 5,942   |
| USMLE Self Assessment Step 1  | 119     |
| USMLE Self Assessment Step 2  | 120     |
| USMLE Self Assessment Step 3  | 135     |

#### Anki Flashcards
Medicine as a whole encompasses a wide range of subjects that medical students and graduates must master in order to practice effectively. This includes a deep understanding of basic medical sciences, clinical knowledge, and clinical skills. The Anki Medical Curriculum flashcards are created and updated by medical students and cover the entirety of this curriculum, addressing subjects such as anatomy, physiology, pathology, pharmacology, and more. These flashcards frequently feature succinct summaries and mnemonics to aid in learning and retention of vital medical concepts.

In our study, we employed the flashcards as a resource for generating question-answer pairs for training purposes. After removing cards that contained images, we utilized OpenAI's GPT-3.5-turbo to rephrase the cards into coherent, contextually relevant question-answer pairs. In general the questions and answers are short and focused, as the flashcards do not allow to add much information.

#### StackExchange 
Our dataset consists of 52,475 question-answer pairs obtained from five Stack Exchange forums related to biomedical sciences and related fields. Specifically, we gathered the data from answers that head at least five upvotes in the forums concering Academia, Bioinformatics, Biology, Fitness, and Health, and matched them to the corresponding question. 

#### WikiDoc
We incorporated medical question-answer pairs extracted from [WikiDoc](https://www.wikidoc.org/index.php/Main_Page), a collaborative platform for medical professionals to share and contribute to up-to-date medical knowledge. The platform has to main subsites, the "Living Textbook" and "Patient Information". The "Living Textbook" contains chapters for various medical specialties, which we crawled. We then used GTP-3.5-Turbo to rephrase the paragraph heading to a question and used the paragraph as answer.  Patient Information is structured differently, in that each section subheading is already a question, making rephrasing them obsolete.  
**Note:** This dataset is still a WIP. While the Q/A pairs from the patient information seems to be mostly correct, the conversion using GPT-3.5-Turbo yielded some unsatisfactory results in  approximately 30% of cases. We are in the process of cleaning this dataset. 

#### USMLE Self assessment
The USMLE (United States Medical Licensing Examination) provides a set of self assessment questions for medical students to evaluate, if they are likels to pass the Exam. For each of the three steps, they provide about 120 questions, which is equal to approximately half the amount of questions a student needs to answer during the real exam. As these questions provide a realsitic knowledge benchmark, covering all relevant medical areas, we use them explicitly for testing our models. 

#### Wikipedia
We have crawled all medical articles on the German and English Wikipedia and are currently in the process of converting them to Q/A pairs. 


### External Datasets
This is a collection of open medical datasets. Please cite the respective source, if you use them in your reserach. 

| Dataset              | n       | link/citation                                                                     |
|----------------------|---------|-----------------------------------------------------------------------------------|
| MEDIQA               | 2208    | https://osf.io/fyg46/?view_only=                                                  |
| CORD-19              | 1056660 | https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge |
| MMMLU                | 3787    | https://github.com/hendrycks/test                                                 |
| MedQA                | 10178   | https://github.com/jind11/MedQA                                                   |
| Pubmed Health Advice | 8676    | https://aclanthology.org/D19-1473/                                                |
| Pubmed Causal        | 2446    | https://aclanthology.org/2020.coling-main.427/                                    |
| ChatDoctor           | 215000  | https://github.com/Kent0n-Li/ChatDoctor                                           |


#### MEDIQA
MEDIQA is a dataset of manually generated, question-driven summaries of multi and single document answers to consumer health questions. 
It is available [here](https://osf.io/fyg46/?view_only=) or, if you want to use our slightly processed version [here](https://huggingface.co/datasets/medalpaca/medical_meadow_mediqa) 
**Citation:**
```
@article{savery2020question,
  title={Question-driven summarization of answers to consumer health questions},
  author={Savery, Max and Abacha, Asma Ben and Gayen, Soumya and Demner-Fushman, Dina},
  journal={Scientific Data},
  volume={7},
  number={1},
  pages={322},
  year={2020},
  publisher={Nature Publishing Group UK London}
}

```

#### COVID-19 Open Research Dataset Challenge (CORD-19)
In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 1,000,000 scholarly articles, including over 400,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. 
The dataset is available [here](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge) or, if you want to use our slightly processed version [here](https://huggingface.co/datasets/medalpaca/medical_meadow_cord19). 
**Citation:**
```
@inproceedings{wang-etal-2020-cord,
    title = "{CORD-19}: The {COVID-19} Open Research Dataset",
    author = "Wang, Lucy Lu  and Lo, Kyle  and Chandrasekhar, Yoganand  and Reas, Russell  and Yang, Jiangjiang  and Burdick, Doug  and Eide, Darrin  and Funk, Kathryn  and Katsis, Yannis  and Kinney, Rodney Michael  and Li, Yunyao  and Liu, Ziyang  and Merrill, William  and Mooney, Paul  and Murdick, Dewey A.  and Rishi, Devvret  and Sheehan, Jerry  and Shen, Zhihong  and Stilson, Brandon  and Wade, Alex D.  and Wang, Kuansan  and Wang, Nancy Xin Ru  and Wilhelm, Christopher  and Xie, Boya  and Raymond, Douglas M.  and Weld, Daniel S.  and Etzioni, Oren  and Kohlmeier, Sebastian",
    booktitle = "Proceedings of the 1st Workshop on {NLP} for {COVID-19} at {ACL} 2020",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.nlpcovid19-acl.1"
}
```

#### Measuring Massive Multitask Language Understanding - Test Set
This is the test set for the Measuring Massive Multitask Language Understanding, which can be downloaded [here](https://people.eecs.berkeley.edu/~hendrycks/data.tar). We also provide a prepared version of this data set [here](https://huggingface.co/datasets/medalpaca/medical_meadow_mmmlu). 

**Citation:**
```
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
#### MedQA
This is the data and baseline source code for the paper: Jin, Di, et al. "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams." 

From https://github.com/jind11/MedQA:
>The data that contains both the QAs and textbooks can be downloaded from [this google drive folder](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing). A bit of details of data are explained as below:
>
> For QAs, we have three sources: US, Mainland of China, and Taiwan District, which are put in folders, respectively. All files for QAs are in jsonl file format, where each line is a data sample as a dict. The "XX_qbank.jsonl" files contain all data samples while we also provide an official random split into train, dev, and test sets. Those files in the "metamap" folders are extracted medical related phrases using the Metamap tool.
>
> For QAs, we also include the "4_options" version in for US and Mainland of China since we reported results for 4 options in the paper.
>
> For textbooks, we have two languages: English and simplified Chinese. For simplified Chinese, we provide two kinds of sentence spliting: one is split by sentences, and the other is split by paragraphs.

If you would like to use the data, please cite the paper. The prepared dataset is available [here](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa)

**Citation:**
```
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
```

####  PubMed Health Advice
This is the dataset use in the paper: Detecting Causal Language Use in Science Findings. 
The prepared dataset is available [here](https://huggingface.co/datasets/medalpaca/medical_meadow_health_advice)

**Citation:**

```
@inproceedings{yu-etal-2019-detecting,
    title = "Detecting Causal Language Use in Science Findings",
    author = "Yu, Bei  and
      Li, Yingya  and
      Wang, Jun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1473",
    doi = "10.18653/v1/D19-1473",
    pages = "4664--4674",
}
```
#### Pubmed Causal
This is the dataset used in the paper: Detecting Causal Language Use in Science Findings.
The prepared dataset is available [here](https://huggingface.co/datasets/medalpaca/medical_meadow_pubmed_causal)

**Citation:**
```
@inproceedings{yu-etal-2019-detecting,
    title = "Detecting Causal Language Use in Science Findings",
    author = "Yu, Bei  and
      Li, Yingya  and
      Wang, Jun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1473",
    doi = "10.18653/v1/D19-1473",
    pages = "4664--4674",
}

```
#### ChatDoctor 
Dataset used in the Paper: "ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge". All data is available [here](https://github.com/Kent0n-Li/ChatDoctor). As it is already in the right format, we provide no prepared version.

Please cite the paper, if you use this dataset. 

**Citation:**

```
@misc{yunxiang2023chatdoctor,
      title={ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge}, 
      author={Li, Yunxiang and Li, Zihan and Zhang, Kai and Dan, Ruilong and Zhang, You},
      year={2023},
      eprint={2303.14070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

#### OpenAssistant Datset
Although, this is not a medical dataste per se, it is a high quality instruction dataset covering multiple languages and was thus also included into our dataset. At this timepoint, the dataset is still in gated access, so we cannot link it directly. Please refer to https://huggingface.co/OpenAssistant, where it will likely be published, once final. 

## Acessing the data
We have parsed all dataset to a uniform format and uploaded them to the hugginface repository. With them you can curate your very own version of the **Medical Meadow** tailored to your specific needs. Check out the data at https://huggingface.co/medalpaca
