import os
import pandas as pd

from src import RAW_METADATA_DATA_DIR, PROCESSED_METADATA_DATA_DIR
from src.utils import print_setup_decorator


@print_setup_decorator("Processing bookcorpus metadata files")
def process_metadata():

    merged_metadata_path = os.path.join(PROCESSED_METADATA_DATA_DIR, 'metadata.csv')

    if not os.path.isfile(merged_metadata_path):

        print(f"Processing and merging metadata")

        ###############################################################################################################
        # This section of code is from https://github.com/jackbandy/bookcorpus-datasheet                              #
        ###############################################################################################################

        bc_books = pd.read_csv(os.path.join(RAW_METADATA_DATA_DIR, 'books_in_bookcorpus.csv'))
        bcopen_books = pd.read_csv(os.path.join(RAW_METADATA_DATA_DIR, '2020-08-27-epub_urls.txt'), header=None, names=['EpubLink'])
        sw_books = pd.read_csv(os.path.join(RAW_METADATA_DATA_DIR, 'smashwords_april_2021_dedup.csv'))
        sw_books['in_smashwords21'] = True

        # merge Smashwords21 with BookCorpusOpen

        # get the "smashwords id" and add a field for bcopen
        bcopen_books['smashwords_id'] = bcopen_books.EpubLink.str.split('/', expand=True)[5]
        bcopen_books['in_bcopen'] = True

        # add "smashwords id" for smashwords21 and merge
        sw_books['smashwords_id'] = sw_books.Link.str.split('/', expand=True)[5]
        sw_books = sw_books.merge(bcopen_books, how='outer', on='smashwords_id')
        sw_books.fillna(value='', inplace=True)

        # partially merge Smashwords21 with original BookCorpus
        bc_books['smashwords_id'] = bc_books.fname.str.replace('.txt', '')
        bc_books['in_bc_books'] = True
        sw_books = sw_books.merge(bc_books, how='left', on='smashwords_id')

        stolen_books = sw_books[
            (sw_books.in_bc_books == True) & (sw_books.Price.str.contains('USD')) & (sw_books.Price != '$0.00 USD')]

        ###############################################################################################################
        # End of code section from https://github.com/jackbandy/bookcorpus-datasheet                                  #
        ###############################################################################################################

        sw_books = sw_books[sw_books['EpubLink'] != ""]
        sw_books = sw_books[sw_books['Categories'].str.contains("Fiction")]
        sw_books = sw_books[~sw_books['smashwords_id'].isin(stolen_books['smashwords_id'])]

        sw_books.to_csv(merged_metadata_path, index=False)

        print(f"Successfully saved processed metadata to {merged_metadata_path}")
    else:
        print(f"Metadata already found in {merged_metadata_path}, skipping this step")


def main():

    process_metadata()


if __name__ == "__main__":
    main()


