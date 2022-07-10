import logging
import os
import time
import sys


def init_logger(args):
    if not os.path.exists("log"):
        os.makedirs("log")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join("log", f"goemotions_{time.strftime('%Y%m%d-%H%M%S')}.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
