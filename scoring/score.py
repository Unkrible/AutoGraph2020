# pylint: disable=logging-fstring-interpolation
"""scoring function for autograph"""

import argparse
import datetime
import os
from os.path import join
import logging
import sys
import time

import yaml
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score

from filelock import FileLock

# Verbosity level of logging.
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
#  VERBOSITY_LEVEL = 'INFO'
VERBOSITY_LEVEL = 'DEBUG'
WAIT_TIME = 30
MAX_TIME_DIFF = datetime.timedelta(seconds=600)
DEFAULT_SCORE = -1
SOLUTION_FILE = 'test_label.tsv'


def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


LOGGER = get_logger(VERBOSITY_LEVEL)


def _here(*args):
    """Helper function for getting the current directory of the script."""
    here_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(join(here_dir, *args))


def _get_solution(solution_dir):
    """Get the solution array from solution directory."""
    solution_file = join(solution_dir, SOLUTION_FILE)
    solution = pd.read_csv(solution_file, sep='\t')
    return solution


def _get_prediction(prediction_dir):
    pred_file = join(prediction_dir, 'predictions')
    return pd.read_csv(pred_file)['label']


def _get_score(solution_dir, prediction_dir):
    """get score"""
    LOGGER.info('===== get solution')
    solution = _get_solution(solution_dir)['label']
    LOGGER.info('===== read prediction')
    prediction = _get_prediction(prediction_dir)
    if solution.shape != prediction.shape:
        raise ValueError(f"Bad prediction shape: {prediction.shape}. "
                         f"Expected shape: {solution.shape}")

    LOGGER.info('===== calculate score')
    LOGGER.debug(f'solution shape = {solution.shape}')
    LOGGER.debug(f'prediction shape = {prediction.shape}')
    score = accuracy_score(solution, prediction)

    def get_df(counter, name):
        counter = {k: v for (k, v) in sorted(counter.items(), key=lambda x: x[0])}
        keys = counter.keys()
        values = counter.values()
        return pd.DataFrame({name: list(values)}, index=keys)

    labels_count = Counter(solution)
    length = len(solution)
    labels = get_df(labels_count, "Label num")
    labels_ratio = get_df({k: labels_count[k] / length for k in labels_count}, "Label ratio")
    errors_count = Counter(solution[solution != prediction])
    errors = get_df(errors_count, "Error")
    errors_ratio = get_df({k: errors_count[k] / labels_count[k] for k in errors_count}, "Error ratio")
    desc = labels.join(labels_ratio).join(errors).join(errors_ratio)
    LOGGER.debug(f"Desc:\n{desc}")

    return score


def _update_score(args, duration):
    score = _get_score(solution_dir=args.solution_dir,
                       prediction_dir=args.prediction_dir)
    # Update learning curve page (detailed_results.html)
    _write_scores_html(args.score_dir)
    # Write score
    LOGGER.info('===== write score')
    write_score(args.score_dir, score, duration)
    LOGGER.info(f"accuracy: {score:.4}")
    return score


def _init_scores_html(detailed_results_filepath):
    html_head = ('<html><head> <meta http-equiv="refresh" content="5"> '
                 '</head><body><pre>')
    html_end = '</pre></body></html>'
    with open(detailed_results_filepath, 'a') as html_file:
        html_file.write(html_head)
        html_file.write("Starting training process... <br> Please be patient. "
                        "Learning curves will be generated when first "
                        "predictions are made.")
        html_file.write(html_end)


def _write_scores_html(score_dir, auto_refresh=True, append=False):
    filename = 'detailed_results.html'
    if auto_refresh:
        html_head = ('<html><head> <meta http-equiv="refresh" content="5"> '
                     '</head><body><pre>')
    else:
        html_head = """<html><body><pre>"""
    html_end = '</pre></body></html>'
    if append:
        mode = 'a'
    else:
        mode = 'w'
    filepath = join(score_dir, filename)
    with open(filepath, mode) as html_file:
        html_file.write(html_head)
        html_file.write(html_end)
    LOGGER.debug(f"Wrote learning curve page to {filepath}")


def write_score(score_dir, score, duration):
    """Write score and duration to score_dir/scores.txt"""
    score_filename = join(score_dir, 'scores.txt')
    with open(score_filename, 'w') as ftmp:
        ftmp.write(f'score: {score}\n')
        ftmp.write(f'Duration: {duration}\n')
    LOGGER.debug(f"Wrote to score_filename={score_filename} with "
                 f"score={score}, duration={duration}")


class IngestionError(Exception):
    """Ingestion error"""


class ScoringError(Exception):
    """scoring error"""


def get_ingestion_info(prediction_dir):
    """get ingestion information"""
    ingestion_info = None
    endfile_path = os.path.join(prediction_dir, 'end.yaml')

    if not os.path.isfile(endfile_path):
        raise IngestionError("[-] No end.yaml exist, ingestion failed")

    LOGGER.info('===== Detected end.yaml file, get ingestion information')
    with open(endfile_path, 'r') as ftmp:
        ingestion_info = yaml.safe_load(ftmp)

    return ingestion_info


def get_ingestion_pid(prediction_dir):
    """get ingestion pid"""
    # Wait 60 seconds for ingestion to start and write 'start.txt',
    # Otherwise, raise an exception.
    wait_time = 60
    startfile = os.path.join(prediction_dir, 'start.txt')
    lockfile = os.path.join(prediction_dir, 'start.txt.lock')

    for i in range(wait_time):
        if os.path.exists(startfile):
            with FileLock(lockfile):
                with open(startfile, 'r') as ftmp:
                    ingestion_pid = ftmp.read()
                    LOGGER.info(
                        f'Detected the start of ingestion after {i} seconds.')
                    return int(ingestion_pid)
        else:
            time.sleep(1)
    raise IngestionError(f'[-] Failed: scoring didn\'t detected the start of'
                         'ingestion after {wait_time} seconds.')


def is_process_alive(ingestion_pid):
    """detect ingestion alive"""
    try:
        os.kill(ingestion_pid, 0)
    except OSError:
        return False
    else:
        return True


def _parse_args():
    # Default I/O directories:
    root_dir = _here(os.pardir)
    default_solution_dir = join(root_dir, "sample_data")
    default_prediction_dir = join(root_dir, "sample_result_submission")
    default_score_dir = join(root_dir, "scoring_output")
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution_dir', type=str,
                        default=default_solution_dir,
                        help=("Directory storing the solution with true "
                              "labels, e.g. adult.solution."))
    parser.add_argument('--prediction_dir', type=str,
                        default=default_prediction_dir,
                        help=("Directory storing the predictions. It should"
                              "contain e.g. [start.txt, adult.predict_0, "
                              "adult.predict_1, ..., end.yaml]."))
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help=("Directory storing the scoring output e.g. "
                              "`scores.txt` and `detailed_results.html`."))
    args = parser.parse_args()
    LOGGER.debug(f"Parsed args are: {args}")
    LOGGER.debug("-" * 50)
    LOGGER.debug(f"Using solution_dir: {args.solution_dir}")
    LOGGER.debug(f"Using prediction_dir: {args.prediction_dir}")
    LOGGER.debug(f"Using score_dir: {args.score_dir}")
    return args


def _init(args):
    if not os.path.isdir(args.score_dir):
        os.mkdir(args.score_dir)
    detailed_results_filepath = join(
        args.score_dir, 'detailed_results.html')
    # Initialize detailed_results.html
    _init_scores_html(detailed_results_filepath)


def _finalize(score, scoring_start):
    """finalize the scoring"""
    # Use 'end.yaml' file to detect if ingestion program ends
    duration = time.time() - scoring_start
    LOGGER.info(
        "[+] Successfully finished scoring! "
        f"Scoring duration: {duration:.2} sec. "
        f"The score of your algorithm on the task is: {score:.6}.")

    LOGGER.info("[Scoring terminated]")


def main():
    """main entry"""
    scoring_start = time.time()
    LOGGER.info('===== init scoring program')
    args = _parse_args()
    _init(args)
    score = DEFAULT_SCORE

    ingestion_pid = get_ingestion_pid(args.prediction_dir)

    LOGGER.info("===== wait for the exit of ingestion.")
    while is_process_alive(ingestion_pid):
        time.sleep(1)

    # Compute/write score
    ingestion_info = get_ingestion_info(args.prediction_dir)
    duration = ingestion_info['ingestion_duration']
    score = _update_score(args, duration)

    _finalize(score, scoring_start)


if __name__ == "__main__":
    main()
