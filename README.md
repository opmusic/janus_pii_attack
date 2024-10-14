## The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks

<p>
    <a href="https://www.python.org/downloads/">
            <img alt="Build" src="https://img.shields.io/badge/3.10-Python-blue">
    </a>
    <a href="https://pytorch.org">
            <img alt="Build" src="https://img.shields.io/badge/1.11-PyTorch-orange">
    </a>
    <a href="https://github.com/pytorch/opacus">
            <img alt="Build" src="https://img.shields.io/badge/1.12-opacus-orange">
    </a>
</p>

This repository contains the official code for our ACM CCS 2024 paper using GPT-2 language models and
Flair Named Entity Recognition (NER) models. It is built upon the github repo: https://github.com/microsoft/analysing_pii_leakage and supports our proposed targeted privacy attack -- Janus attack.


## Publication

> **The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks.**
> Xiaoyi Chen and Siyuan Tang and Rui Zhu (equal contribution), Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang.
> ACM Conference on Computer and Communications Security (CCS'24). Salt Lake City, USA.
>
> [![arXiv](https://img.shields.io/badge/arXiv-2310.15469-green)](https://arxiv.org/abs/2310.15469)


## Build & Run

We recommend setting up a conda environment for this project.
```shell
$ conda create -n pii-leakage python=3.10
$ conda activate pii-leakage
$ pip install -e .
```

## Usage

We explain the following functions. The scripts are in the ```./examples``` folder and
run configurations are in the ```./configs``` folder.
* **Pretrain**: Simulate the pretraining process of language models through continual learning.
* **Attack**: Implement the Janus attack on the languagee models
* **Evaluation**: Implement the Janus attack on the language models


## Pretrain

We demonstrate how to simulate the pretraining of ```GPT-2``` ([Huggingface](https://huggingface.co/gpt2)) models on the [ECHR](https://huggingface.co/datasets/ecthr_cases) and [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) datasets.

Edit your own path for the original model and saved pretrained model in the ```../configs/targted-attack/echr-gpt2-janus-pretrain.yml``` The default output folder is your current folder.

```shell
$ python janus_pretrain.py --config_path ../configs/targted-attack/echr-gpt2-janus-pretrain.yml
```

## Attack

Edit the ```model_ckpt``` attribute in the ```../configs/targted-attack/echr-gpt2-janus-attack.yml``` file to point to the location of the saved pretrained model. Edit the ```root``` attribute to specify the output folder of the attacked model.

```shell
$ python janus_attack.py --config_path ../configs/targted-attack/echr-gpt2-janus-attack.yml
```

## Evaluation

Edit the ```model_ckpt``` attribute in the ```../configs/targted-attack/echr-gpt2-janus-eval.yml``` file to point to the location of the model you want to evaluate.

```shell
$ python evaluate.py --config_path ../configs/targted-attack/echr-gpt2-janus-eval.yml
```

## Datasets

The provided ECHR dataset wrapper already tags all PII in the dataset.
The PII tagging is done using the Flair NER modules and can take several hours depending on your setup, but is a one-time operation
that will be cached in subsequent runs.


## Fine-Tuned Models

Currently, we do not provide the fine-tuned models in the repo. If you have further questions, please contact the authors.

## Citation

Please consider citing our paper if you found our work useful.
