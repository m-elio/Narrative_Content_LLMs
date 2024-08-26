import os
import re
import json

from tqdm import tqdm

from src import RAW_ADAPTATION_DATA_DIR, PROCESSED_ADAPTATION_DATA_DIR
from src.utils import print_setup_decorator


@print_setup_decorator("Processing raw Gutenberg Fiction corpus")
def process_fiction_gutenberg_ds():

    original = 0
    processed = 0
    processed_paragraphs = 0

    fiction_gutenberg_processed_path = os.path.join(PROCESSED_ADAPTATION_DATA_DIR, 'fiction_gutenberg_processed.jsonl')

    if not os.path.isfile(fiction_gutenberg_processed_path):

        print(f"Processing Gutenberg dataset")

        with open(os.path.join(RAW_ADAPTATION_DATA_DIR, 'fiction_gutenberg.jsonl'), 'r', encoding='utf8') as f_read:

            with open(fiction_gutenberg_processed_path, 'w', encoding='utf8') as f_out:

                for line in tqdm(f_read.readlines()):

                    data = json.loads(line)

                    original += 1

                    # only keep books which author year of birth is after 1800
                    if data['authoryearofbirth'] >= 1800:

                        processed_text = data['text']

                        # split on double new line which usually means new paragraph in this corpus
                        processed_text = processed_text.split('*** END OF THE PROJECT GUTENBERG EBOOK')[0].split('\n\n')

                        # replace new lines with whitespace and remove multiple whitespaces (avoid strings such as "" or "\n")
                        processed_text = [re.sub(' +', ' ', x.replace('\n', ' ').strip()) for x in processed_text if len(x) > 2]
                        final_processed_text = [x for x in processed_text if len(x) > 50 and len(x) < 10000]

                        if len(final_processed_text) == 0:
                            continue

                        processed += 1
                        processed_paragraphs += len(final_processed_text)

                        for x in final_processed_text:
                            text_data = {}
                            text_data['text'] = x
                            json.dump(text_data, f_out)
                            f_out.write('\n')

        print(f"Successfully saved processed dataset to {fiction_gutenberg_processed_path}")
        print(f"Number of original instances: {original}")
        print(f"Number of processed instances: {processed}")
        print(f"Number of processed paragraphs: {processed_paragraphs}")
    else:
        print(f"Dataset already found in {fiction_gutenberg_processed_path}, skipping this step")


@print_setup_decorator("Processing raw bookcorpus dataset")
def process_fiction_bookcorpus():

    original = 0
    processed = 0
    processed_paragraphs = 0

    bookcorpus_processed_path = os.path.join(PROCESSED_ADAPTATION_DATA_DIR, 'fiction_bookcorpus_processed.jsonl')

    if not os.path.isfile(bookcorpus_processed_path):

        print(f"Processing bookcorpus dataset")

        with open(os.path.join(RAW_ADAPTATION_DATA_DIR, 'fiction_bookcorpus.jsonl'), 'r', encoding='utf8') as f_read:

            with open(bookcorpus_processed_path, 'w', encoding='utf8') as f_out:

                for line in tqdm(f_read.readlines()):

                    data = json.loads(line)

                    raw_text = data['text']

                    processed_text = re.split('\n\n', raw_text)

                    # replace new lines with whitespace and remove multiple whitespaces (avoid strings such as "" or "\n")
                    processed_text = [re.sub(' +', ' ', x.replace('\n', ' ').strip()) for x in processed_text if len(x) > 2]
                    final_processed_text = [x for x in processed_text if len(x) > 50 and len(x) < 10000]

                    original += 1

                    if len(final_processed_text) == 0:
                        continue

                    processed += 1
                    processed_paragraphs += len(final_processed_text)

                    for x in final_processed_text:
                        text_data = {}
                        text_data['text'] = x
                        json.dump(text_data, f_out)
                        f_out.write('\n')

        print(f"Successfully saved processed dataset to {bookcorpus_processed_path}")
        print(f"Number of original instances: {original}")
        print(f"Number of processed instances: {processed}")
        print(f"Number of processed paragraphs: {processed_paragraphs}")
    else:
        print(f"Dataset already found in {bookcorpus_processed_path}, skipping this step")


@print_setup_decorator("Combining Bookcorpus and Gutenberg datasets")
def combine_bookcorpus_gutenberg():

    original = 0

    bookcorpus_gutenberg_processed_path = os.path.join(PROCESSED_ADAPTATION_DATA_DIR, 'fiction_bookcorpus_gutenberg_processed.jsonl')

    if not os.path.isfile(bookcorpus_gutenberg_processed_path):
        
        with open(bookcorpus_gutenberg_processed_path, 'w', encoding='utf8') as f_out:

            print(f"Copying Bookcorpus instances")

            with open(os.path.join(PROCESSED_ADAPTATION_DATA_DIR, 'fiction_bookcorpus_processed.jsonl'), 'r', encoding='utf8') as f_read:


                    for line in tqdm(f_read.readlines()):

                        data = json.loads(line)
                        json.dump(data, f_out)
                        f_out.write('\n')
                        original += 1

            print(f"Copying Gutenberg instances")

            with open(os.path.join(PROCESSED_ADAPTATION_DATA_DIR, 'fiction_gutenberg_processed.jsonl'), 'r', encoding='utf8') as f_read:

                    for line in tqdm(f_read.readlines()):

                        data = json.loads(line)
                        json.dump(data, f_out)
                        f_out.write('\n')
                    original += 1

        print(f"Successfully saved combined dataset to {bookcorpus_gutenberg_processed_path}")
        print(f"Number of total instances: {original}")
    else:
        print(f"Dataset already found in {bookcorpus_gutenberg_processed_path}, skipping this step")


def main():

    process_fiction_gutenberg_ds()
    process_fiction_bookcorpus()
    combine_bookcorpus_gutenberg()


if __name__ == "__main__":
    main()
