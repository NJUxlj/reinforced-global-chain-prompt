import logging
import os
import sys

names = set()


def __setup_custom_logger(name: str) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    names.add(name)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    ## TODO
    file_handler = logging.FileHandler(name + ".txt")
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    ## TODO
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    if name in names:
        return logging.getLogger(name)
    else:
        return __setup_custom_logger(name)
