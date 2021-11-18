import os
from inference import prepare_hyperparameters

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    BATCH_SIZE = 64
    GPUS_LIST = [2]
    MODEL_HYPERPARAMETERS = prepare_hyperparameters()

    JSON_AS_ASCII = False
