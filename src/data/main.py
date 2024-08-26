from src.data.adaptation.process_books import main as process_adaptation

from src.data.metadata.get_metadata import main as get_metadata
from src.data.metadata.process_metadata import main as process_metadata

from src.data.tuning.get_ds import main as get_tuning
from src.data.tuning.process_ds import main as process_tuning
from src.data.tuning.create_prompt_rank_ds import main as create_prompt_rank_ds


def main():

    get_metadata()
    process_metadata()

    #######################################################################################################################################################
    # The filtering script should be added here, in particular:
    #
    # - For the Gutenberg dataset it should select only fiction works
    # - For the BookCorpus dataset it should perform the filtering operations described in the paper using the metadata downloaded in the previous step
    #
    # Once this has been done, it is also possible to uncomment the process_adaptation function
    #######################################################################################################################################################

    # process_adaptation()

    get_tuning()
    process_tuning()
    create_prompt_rank_ds()


if __name__ == "__main__":
    main()
