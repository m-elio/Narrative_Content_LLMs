import os
import yaml
import shutil

from src import MODELS_DIR
from src.train import train


def main():

    with open('parameters/parameters_train.yaml') as file:
        parameters = yaml.load(file, yaml.FullLoader)

    new_model_name = parameters['new_model_name']
    save_directory = os.path.join(MODELS_DIR, new_model_name)

    os.makedirs(save_directory, exist_ok=True)
    shutil.copy('parameters/parameters_train.yaml', os.path.join(save_directory, 'parameters/parameters_train.yaml'))
    
    train(parameters, save_directory)


if __name__ == "__main__":
    main()
