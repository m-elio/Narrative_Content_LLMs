import os
import yaml
import torch
import datasets

from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers.utils import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed

from src import DATA_DIR, HF_CACHE_DIR
from src.utils.utils import POSSIBLE_DTYPES


def prompt_accuracy(max_seq_len):

    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")

    with open('parameters/parameters_inference.yaml') as file:
        parameters_inf = yaml.load(file, yaml.FullLoader)

    model_dir = parameters_inf["model_dir"]
    dataset_path = 'tuning/processed/writing_prompts_test_hf_processed_prompt_accuracy.jsonl'

    parameters_path = os.path.join(model_dir, 'parameters.yaml')

    if os.path.isfile(parameters_path):
        with open(parameters_path) as file:
            parameters_train = yaml.load(file, yaml.FullLoader)
    else:
        parameters_train = {}
    
    model_path = model_dir

    pretrained_parameters = parameters_inf.get("pretrained_parameters", {})
    generation_parameters = parameters_inf.get("generation_parameters", {})
    batch_size = parameters_inf.get("batch_size", 8)

    padding_side = parameters_inf.get("padding_side", None)

    logger.info(f"Padding side: {padding_side}")
    logger.info(f"Batch size: {batch_size}")
    
    if "torch_dtype" in pretrained_parameters:
        pretrained_parameters["torch_dtype"] = POSSIBLE_DTYPES[pretrained_parameters["torch_dtype"]]
    
    if "torch_dtype" in generation_parameters:
        generation_parameters["torch_dtype"] = POSSIBLE_DTYPES[generation_parameters["torch_dtype"]]

    training_parameters = parameters_train.get("training_arguments", {})

    logger.info("Setting seed")

    seed = training_parameters.get("seed", 42)

    set_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=None,
        device_map="auto",
        cache_dir=HF_CACHE_DIR,
        **pretrained_parameters
    )

    model = model.eval()

    logger.info('Model loaded.')
    logger.info(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    dataset_extension = dataset_path.split('.')[-1]

    if dataset_extension == "jsonl":
        dataset_extension = "json"

    # https://github.com/huggingface/datasets/issues/1785#issuecomment-1305872400
    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
    dataset = load_dataset(dataset_extension, data_files=os.path.join(DATA_DIR, dataset_path))['train']

    logger.info(generation_parameters)
    logger.info('Data loaded.')
    logger.info(dataset_path)

    device = "cuda"
    
    d_len = len(dataset['sample_all'])
    hit = 0
    total = 0

    logger.info(f'Evaluating with max seq length {max_seq_len}')

    loss_fct = CrossEntropyLoss(reduction="none")

    max_tokenized_len = max_seq_len
    batch_size = 8

    tens = dataset['sample_all']
    prompts = dataset['sample_prompts']
    
    loss_fct = CrossEntropyLoss(reduction="none")

    for i in range(d_len):

        logls = []

        sample_prompts = prompts[i]
        sample_ten = tens[i]

        prompt_lens = [len(tokenizer(p, truncation=True if max_tokenized_len else False, max_length=max_seq_len).input_ids) for p in sample_prompts]
        encodings = tokenizer(sample_ten, padding=True, truncation=True if max_tokenized_len else False, max_length=max_seq_len, return_tensors="pt")

        encoded_texts = encodings["input_ids"].to(device)
        attn_mask = encodings["attention_mask"].to(device)

        labels = encoded_texts

        with torch.no_grad():
            out_logits = model(encoded_texts, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        log_probs = loss_fct(shift_logits.transpose(1, 2), shift_labels)

        for j in range(log_probs.shape[0]):
            shift_att = shift_attention_mask_batch[j, prompt_lens[j]:]
            lp = log_probs[j, prompt_lens[j]:]
            lp = lp[torch.gt(shift_att, 0)]
            logls.append(lp.float().mean().item())
        
        print(logls)  
        if logls[0] == min(logls):
            hit += 1
        total += 1

        print('Prompt Ranking accuracy: {}'.format(
            hit / total
        ))


if __name__ == '__main__':
    
    max_seq_len = 2048

    prompt_accuracy(max_seq_len)