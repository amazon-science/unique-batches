# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import logging


def get_logger(root: str = None) -> logging.Logger:
    """Gets configured logger

    Args:
        root (str, optional): root of the logger. Defaults to None.

    Returns:
        logging.Logger: logger
    """

    logger = logging.getLogger(root)
    logger.setLevel(logging.INFO)

    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.INFO)
    consolehandler.setFormatter(format)
    logger.addHandler(consolehandler)

    # logger.propagate = False

    return logger
