# Adapting Large Language Models to Narrative Content

Welcome! This is the codebase for the paper "Adapting Large Language Models to Narrative Content" accepted at the [CREAI 2024 Workshop](https://creai.github.io/creai2024/) of the [ECAI 2024](https://www.ecai2024.eu/) conference. 
This repo contains the code to reproduce the experiments described in the work, in particular the code does the following:

- Obtain the dataset
- Fine-tune a Large Language Model for story generation
- Evaluate the quality of the generated stories using Perplexity and Prompt Ranking Accuracy

## Requirements

To reproduce the experiments, create a Python 3.10 virtual environment and install the requirements listed in the [requirements file](requirements.txt).

Alternatively, you can also build a container, for the experiments, Singularity 3.9 Pro was used starting from this [Docker image](https://hub.docker.com/layers/nvidia/cuda/12.1.0-cudnn8-devel-ubuntu20.04/images/sha256-da5f69611ae7526fbd23f8f8edb06d1818a782f1bbed7b6508efca1cd8d87777?context=explore).

Finally, to smoothly run the scripts, **set the working directory to the root of this project**.

## Dataset

The first step is to obtain the datasets and the base models that were used. Since we cannot distributed the datasets used for the adaptation step, you should obtain a copy of the Gutenberg and BookCorpus datasets on your own. 

For Gutenberg, we recommend using this same [GitHub repository] that we also used. Then, create a jsonl file where every instance contains the field name and the metadata fields extracted from the repository. The most important attributes that should be in this file are the following: 'authoryearofbirth' and 'text'. 
We provide the list of ids of the Gutenberg works that were used for the experiments.

For BookCorpus, we cannot provide a source to retrieve the dataset. 

Please note that in both cases you will need to provide ".jsonl" files filtered to only contain fiction books. Refer to the publication for more information about the filtering procedure that was applied.
In [this directory](/data/adaptation/raw) we provide an example of the expected files.

For WritingPrompts, we provide the code to download the dataset and process it.

After adding the filtered adaptation datasets to the [raw adaptation directory](/data/adaptation/raw), execute the following command:

    python3.10 -m src.data.main

This will execute all additional processing operations required (e.g. formatting the WritingPrompts dataset to follow the Alpaca format).

## Training

To fine-tune a model, modify the [train parameters file](parameters/parameters_train.yaml). By default, this file contains an example of the parameters used for one of the models. The most important parameters to set are:

- **original_model_path**: the path to the model to be fine-tuned
- **new_model_path**: the path where the fine-tuned model will be saved
- **dataset_path**: the path to the dataset to use (this depends on the training step that is being performed, either domain adaptation or instruction-tuning)

You can now run the fine-tune process by executing the following command:

    python3.10 -m main_train

The script can also be easily launched with the [accelerate](https://huggingface.co/docs/accelerate/index) library as well in combination with deepspeed / fsdp for parallelism.

## Evaluation

The evaluation procedure follows a similar pipeline to the one used for fine-tuning. Modify the [inference parameters file](parameters/parameters_inference.yaml) according to your needs. By default, this file contains an example of the parameters used for one of the models. The most important parameter to set is the following:

- **model_dir**: the path to the fine-tuned model directory saved by the HuggingFace trainer

You can now run the evaluation process for the **Perplexity** and **Prompt Rank** metrics by executing the following command:

    python3.10 -m compute_ppl
    python3.10 -m compute_prompt_rank

The final result for both metrics will be printed on the terminal.

To modify the *max sequence length*, change the value of the variable **max_seq_length** that can be found at the top of the main of the two scripts.
