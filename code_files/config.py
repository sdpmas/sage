import os
import sys
sys.path.append(".")
import logging
import logging.config

from code_files import utils
# print(utils)
# Directories
BASE_DIR = os.getcwd()  # project root
APP_DIR = os.path.dirname(__file__)  # app root
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Create dirs
utils.create_dirs(LOGS_DIR)
utils.create_dirs(DATA_DIR)
utils.create_dirs(MODEL_DIR)

# Loggers
log_config = utils.load_json(
    filepath=os.path.join(BASE_DIR, 'logging.json'))
logging.config.dictConfig(log_config)
logger = logging.getLogger('logger')
