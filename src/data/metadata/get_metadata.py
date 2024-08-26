import os

from urllib.request import urlretrieve

from src import RAW_METADATA_DATA_DIR
from src.utils import print_setup_decorator


def download_file(path: str, url: str):
    if not os.path.isfile(path):
        print(f"Downloading {os.path.basename(path)}")
        urlretrieve(url, filename=path)
        print(f"Successfully saved metadata to {path}")
    else:
        print(f"Metadata already downloaded to {path}, skipping this step")


@print_setup_decorator("Downloading BookCorpus metadata from GitHub")
def download_metadata():

    books_in_bookcorpus_metadata_path = os.path.join(RAW_METADATA_DATA_DIR, 'books_in_bookcorpus.csv')
    download_file(books_in_bookcorpus_metadata_path, 'https://raw.githubusercontent.com/jackbandy/bookcorpus-datasheet/main/data/BookCorpus/books_in_bookcorpus.csv')

    epub_urls_metadata_path = os.path.join(RAW_METADATA_DATA_DIR, '2020-08-27-epub_urls.txt')
    download_file(epub_urls_metadata_path, 'https://raw.githubusercontent.com/jackbandy/bookcorpus-datasheet/main/data/BookCorpusOpen/2020-08-27-epub_urls.txt')

    smashwords_data = os.path.join(RAW_METADATA_DATA_DIR, 'smashwords_april_2021_dedup.csv')
    download_file(smashwords_data, 'https://raw.githubusercontent.com/jackbandy/bookcorpus-datasheet/main/data/Smashwords21/smashwords_april_2021_dedup.csv')


def main():

    download_metadata()


if __name__ == "__main__":
    main()