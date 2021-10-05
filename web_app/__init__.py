import logging
import os
from flask import Flask
from logging.handlers import RotatingFileHandler

from base import get_device, gpus_to_use, init_random_seed_torch
from inference import model_and_tokenizer_initialize
from web_app.config import Config


def create_app(config_class=Config):
    init_random_seed_torch(1995)

    app = Flask(__name__)
    app.config.from_object(config_class)
    gpus_to_use(app.config.get('GPUS_LIST'))
    app.device = get_device(app.config.get('GPUS_LIST'))
    app.model_hyperparameters = app.config.get('MODEL_HYPERPARAMETERS')
    app.model, app.tokenizer = model_and_tokenizer_initialize(app.device, app.model_hyperparameters)

    from web_app.main import bp as main_bp
    app.register_blueprint(main_bp)

    if not app.debug and not app.testing:

        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/punct_restoration_model.log',
                                           maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('Punctuation restoration')

    return app


