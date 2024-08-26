import os
import json

from datasets import load_dataset

from src import RAW_TUNING_DATA_DIR
from src.utils import print_setup_decorator


@print_setup_decorator("Downloading WritingPrompts dataset from HuggingFace")
def download_hf_writing_prompts():

    for split in ["train", "validation", "test"]:

        split_prompts_ds_path = os.path.join(RAW_TUNING_DATA_DIR, f'writing_prompts_{split}_hf.jsonl')

        if not os.path.isfile(split_prompts_ds_path):

            print(f"Downloading HF WritingPrompts {split} dataset")

            ds_hf = load_dataset('euclaise/writingprompts', split=split)

            with open(split_prompts_ds_path, 'w', encoding='utf8') as f_out:

                for prompt in ds_hf:
                    json.dump(prompt, f_out)
                    f_out.write('\n')

            print(f"Successfully saved dataset to {split_prompts_ds_path}")
        else:
            print(f"Dataset already downloaded to {split_prompts_ds_path}, skipping this step")


def main():

    download_hf_writing_prompts()


if __name__ == "__main__":
    main()
