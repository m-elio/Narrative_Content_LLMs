###########################################################################################################
# NOTE: This script is a modified version of the original evaluate library perplexity                     #
# implementation + the implementation released here: https://github.com/calclavia/story-generation        #
###########################################################################################################

import os
import yaml
import torch
import datasets
import numpy as np

from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed

from src import DATA_DIR
from src.utils.utils import POSSIBLE_DTYPES, print_on_main, available_formats


def compute_word_level_ppl(target_tokens, log_probs, tokenizer, attn_mask):

    word_log_probs = []
    cur_log_probs = []
    new_add = ''
    i = 0
    start = False
    
    target_tokens = target_tokens[torch.gt(attn_mask, 0)]
    log_probs = -log_probs[torch.gt(attn_mask, 0)]

    target_tokens_converted = tokenizer.convert_ids_to_tokens(target_tokens)

    for token, log_prob in zip(target_tokens_converted, log_probs):

        if token.startswith('▁'):
            token = token.replace('▁', ' ', 1)

        if token == '<0x0A>':
            new_add += '\n'
        
        new_add += token
        cur_log_probs.append(log_prob)

        if not start:
            start = 'Response:\n' in new_add
            if start:
                cur_log_probs = []
                new_add = ''
            continue
        
        text = new_add
        tokens = text.strip().split(' ')

        if len(tokens) > i + 1:
            # Token length changed, which means new word has been added.
            # Grab all but the last prob (excluding the unformed next word)
            word_log_probs.append(sum(cur_log_probs[:-1]))
            cur_log_probs = cur_log_probs[-1:]
            i += 1
    
    word_log_probs.append(sum(cur_log_probs))
    word_log_probs = torch.Tensor(word_log_probs).to("cuda")
    word_ppl = torch.exp(-word_log_probs.mean()).item()
    
    if word_ppl == float('inf'):
        raise Exception('Infinite PPL')

    if word_ppl > 1000:
        raise Exception('Large PPL')

    return word_ppl


def _compute(
        predictions, model_id, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None
    ):

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        if add_start_token and max_length:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        word_ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in range(0, len(encoded_texts), batch_size):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            log_probs = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            perplexity_batch = torch.exp((log_probs * shift_attention_mask_batch).sum(1) / shift_attention_mask_batch.sum(1))

            ppls += perplexity_batch.tolist()

            for i in range(len(encoded_batch)):

                try:

                    sample_labels = shift_labels[i]
                    sample_log_probs = log_probs[i]
                    sample_attn_mask = shift_attention_mask_batch[i]
                    word_ppl = compute_word_level_ppl(sample_labels, sample_log_probs.float(), tokenizer, sample_attn_mask)
                    word_ppls.append(word_ppl)
                
                except Exception as e:
                        print('Skipping anomaly.')
                        print(e)

        return {"perplexities": ppls, "word_perplexities": word_ppls, "mean_perplexity": np.mean(ppls), "mean_word_perplexity": np.mean(word_ppls)}


if __name__ == "__main__":
    
    max_length = 2048

    with open('parameters/parameters_inference.yaml') as file:
        parameters_inf = yaml.load(file, yaml.FullLoader)

    model_dir = parameters_inf["model_dir"]
    dataset_path = parameters_inf["dataset_path"]
    
    model_path = model_dir

    pretrained_parameters = parameters_inf.get("pretrained_parameters", {})
    generation_parameters = parameters_inf.get("generation_parameters", {})
    batch_size = parameters_inf.get("batch_size", 32)

    padding_side = parameters_inf.get("padding_side", None)

    print_on_main(f"Padding side: {padding_side}")
    print_on_main(f"Batch size: {batch_size}")
    
    if "torch_dtype" in pretrained_parameters:
        pretrained_parameters["torch_dtype"] = POSSIBLE_DTYPES[pretrained_parameters["torch_dtype"]]
    
    if "torch_dtype" in generation_parameters:
        generation_parameters["torch_dtype"] = POSSIBLE_DTYPES[generation_parameters["torch_dtype"]]

    print_on_main("Setting seed")

    seed = 42
    set_seed(seed)
    
    dataset_extension = dataset_path.split('.')[-1]

    if dataset_extension == "jsonl":
        dataset_extension = "json"

    # https://github.com/huggingface/datasets/issues/1785#issuecomment-1305872400
    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
    dataset = load_dataset(dataset_extension, data_files=os.path.join(DATA_DIR, dataset_path))['train']

    input_format = 'raw'
    instruction_format_parameters = 'prompt'
    instruction_formatter = available_formats[input_format](**instruction_format_parameters)
    formatting_func = lambda x: instruction_formatter.get_prompt(x, is_train=False)

    print_on_main(formatting_func(dataset[0]))

    print_on_main("Applying input formatter to dataset")
    dataset = dataset.map(lambda x: {'input': formatting_func(x)})
    
    print('Data loaded.')

    device = "cuda"

    print(model_path)
    print(max_length)

    we = [x + y for x,y in zip(dataset['input'], dataset['answer'])]

    a = _compute(we, model_path, batch_size, True, device, max_length)

    print(a["mean_word_perplexity"], a["mean_perplexity"])