import os
import numpy as np
import time
import logging
import sys
import subprocess
import shlex
from shutil import copyfile
import json
from threading import Timer
from os import listdir
from os.path import isfile, join


def minibatches(data_generator, minibatch_size):
    """
    Args:
        data_generator: generator of (img, formulas) tuples
        minibatch_size: (int)

    Returns:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data_generator:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def run(cmd, timeout_sec):
    """Run cmd in the shell with timeout"""
    proc = subprocess.Popen(cmd, shell=True)
    def kill_proc(p): return p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()


def get_logger(filename):
    """Return instance of logger"""
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def init_dir(dir_name):
    """Creates directory if it does not exists"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def init_file(path_file, mode="a"):
    """Makes sure that a given file exists"""
    with open(path_file, mode) as f:
        pass


def get_files(dir_name):
    files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    return files


def delete_file(path_file):
    try:
        os.remove(path_file)
    except Exception:
        pass


def write_answers(references, hypotheses, dir_name):
    """Writes text answers in files.

    One file for the reference, one file for each hypotheses

    Args:
        references: list of list (one reference)
        hypotheses: list of list (multiple hypotheses)
        dir_name: (string) path where to write results

    Returns:
        file_names: list of the created files

    """
    def write_file(file_name, list_of_list):
        print("writting file ", file_name)
        with open(file_name, "w") as f:
            for line in list_of_list:
                f.write(str(line) + "\n")

    init_dir(dir_name)
    file_names = [dir_name + "ref.csv", dir_name + "hyp.csv"]
    assert len(references) == len(hypotheses)
    write_file(file_names[0], references)
    write_file(file_names[1], hypotheses)

    return file_names
