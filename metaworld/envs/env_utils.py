import logging
import os
import os.path as osp
from datetime import datetime


def setup_logger():
    if len(logging.root.handlers) == 0:
        current_datetime = datetime.now()
        current_datetime = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')
        fmt = '[%(asctime)s %(levelname)s %(name)s]: %(message)s'
        filename = osp.join('logs', f'logfile-{current_datetime}.log')
        os.makedirs('logs', exist_ok=True)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(filename=filename)
        logging.basicConfig(format=fmt,
                            level=logging.INFO,
                            handlers=[console_handler, file_handler])
        logger = logging.getLogger(__name__)
        logger.info(f'Setup logger, logs will be saved to {filename}.')


def get_logger(name):
    setup_logger()
    return logging.getLogger(name)
