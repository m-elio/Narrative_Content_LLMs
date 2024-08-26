import os
import json
import numpy as np

from tqdm import tqdm
from transformers.trainer_utils import set_seed

from src import PROCESSED_TUNING_DATA_DIR


def main():

    set_seed(42)

    num_samples = 1000
    wrong_sample_size = 9

    original_test_ds_path = os.path.join(PROCESSED_TUNING_DATA_DIR, 'writing_prompts_test_hf_processed.jsonl')
    new_test_ds_path = os.path.join(PROCESSED_TUNING_DATA_DIR, 'writing_prompts_test_hf_processed_prompt_accuracy.jsonl')

    prompts = []
    stories = []

    with open(original_test_ds_path, 'r', encoding='utf-8') as f:

        for line in tqdm(f.readlines()):

            data = json.loads(line)

            prompts.append(data['prompt'])
            stories.append(data['answer'])

    all_idxs = np.arange(len(prompts))
    random_story_idxs = np.random.choice(all_idxs, num_samples, replace=False)

    with open(new_test_ds_path, 'w', encoding='utf-8') as f:

        for random_story_idx in random_story_idxs:
            
            all_possible_prompt_idxs = all_idxs.copy()
            all_possible_prompt_idxs = np.hstack([all_possible_prompt_idxs[0:random_story_idx], all_possible_prompt_idxs[random_story_idx:len(all_possible_prompt_idxs)]])

            while True:

                random_prompt_idxs = np.random.choice(all_possible_prompt_idxs, wrong_sample_size, replace=False)
            
                correct_prompt = prompts[random_story_idx]
                story = stories[random_story_idx]

                sample_prompts = []
                sample_all = []

                sample_prompts.append(correct_prompt)
                sample_all.append(correct_prompt + story)

                for x in random_prompt_idxs:
                    sample_prompts.append(prompts[x])
                    sample_all.append(prompts[x] + story)
                
                if len(set(sample_prompts)) == 10:
                    break
            
            data = {}
            data['sample_all'] = sample_all
            data['sample_prompts'] = sample_prompts

            json.dump(data, f)
            f.write('\n')


if __name__ == "__main__":

    main()