import os
import re
import json

from tqdm import tqdm

from src import RAW_TUNING_DATA_DIR, PROCESSED_TUNING_DATA_DIR
from src.utils import print_setup_decorator


def wp_preprocess(text):
    text = text.replace('<newline>', '\n')
    text = text.replace('``', '"')
    text = text.replace("''", '"')
    text = re.sub(' +', ' ', text)
    text = re.sub(' (\'|\.|\,|\:|\?|\!|;)', '\g<1>', text)
    text = re.sub('" (.*) "', '"\g<1>"', text)
    text = text.replace(" n't", "n't")
    return text


@print_setup_decorator("Processing raw Writing Prompts dataset")
def process_fiction_writing_prompts_ds():

    for split in ["train", "validation", "test"]:

        instances = 0

        writing_prompts_processed_path = os.path.join(PROCESSED_TUNING_DATA_DIR, f'writing_prompts_{split}_hf_processed.jsonl')

        if not os.path.isfile(writing_prompts_processed_path):

            with open(os.path.join(RAW_TUNING_DATA_DIR, f'writing_prompts_{split}_hf.jsonl'), 'r', encoding='utf8') as f_read:

                with open(writing_prompts_processed_path, 'w', encoding='utf8') as f_out:

                    for line in tqdm(f_read.readlines()):

                        data = json.loads(line)
                        new_data = {}

                        prompt = data['prompt']
                        prompt = re.sub('\[ (.*) \]', '', prompt)
                        prompt = wp_preprocess(prompt).strip()

                        story = wp_preprocess(data['story']).strip()

                        instances += 1

                        story_prompt = story if split == "train" else ""
                            
                        instruction_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n" \
                        "### Instruction:\nWrite a story for the writing prompt provided as input\n\n" \
                        f"### Input:\n{prompt}\n\n" \
                        f"### Response:\n{story_prompt}"

                        new_data['prompt'] = instruction_prompt
                        new_data['answer'] = story

                        json.dump(new_data, f_out)
                        f_out.write('\n')

            print(f"Successfully saved dataset to {writing_prompts_processed_path}, total number of instances: {instances}")
        else:
            print(f"Dataset already downloaded to {writing_prompts_processed_path}, skipping this step")


def main():

    process_fiction_writing_prompts_ds()


if __name__ == "__main__":
    main()
