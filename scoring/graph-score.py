#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/4/30 19:05
# @Author:  Mecthew

import os
import sys
import re
from datetime import datetime
import numpy as np


class ScoreTuple:
    def __init__(self, dataset_name, score_list):
        self.name = dataset_name
        self.dataset_score_list = score_list

    def __str__(self):
        return "%-18s\tmean %.6f,  max %.6f,  min %.6f,  std %.6f,  num %d" % (self.name, np.mean(self.dataset_score_list), np.max(self.dataset_score_list),
                                                                               np.min(self.dataset_score_list), np.std(self.dataset_score_list), len(self.dataset_score_list))


def read_score_of_dir(dir_path):
    score_list = []
    time_cost_patten = re.compile(r'Scoring duration: ([0-9e+-\\.]+) sec.')
    score_patten = re.compile(r'The score of your algorithm on the task is: ([0-9\\.]+).')

    for file_name in os.listdir(dir_path):
        if file_name.endswith(".log"):
            file_path = os.path.join(dir_path, file_name)
            score, time_duration = None, None
            for line in open(file_path, 'r', encoding="utf8"):
                if line.strip().startswith(datetime.now().year.__str__()) and \
                        (score is None or time_duration is None):
                    try:
                        time_duration = time_cost_patten.findall(line.strip())[0]
                    except Exception as e:
                        pass
                    try:
                        score = score_patten.findall(line.strip())[0]
                    except Exception as e:
                        pass
            score_list.append((file_name, time_duration, score))
        elif os.path.isdir(os.path.join(dir_path, file_name)):
            child_dir_path = os.path.join(dir_path, file_name)
            for subfile in os.listdir(child_dir_path):
                if subfile.endswith(".log"):
                    file_path = os.path.join(child_dir_path, subfile)
                    score, time_duration = None, None
                    for line in open(file_path, 'r', encoding="utf8"):
                        if line.strip().startswith(datetime.now().year.__str__()) and \
                                (score is None or time_duration is None):
                            try:
                                time_duration = time_cost_patten.findall(line.strip())[0]
                            except Exception as e:
                                pass
                            try:
                                score = score_patten.findall(line.strip())[0]
                            except Exception as e:
                                pass
                    score_list.append((os.path.join(file_name, subfile), time_duration, score))

    return score_list


def main(argv):
    score_list = read_score_of_dir(argv[1])
    mean_score_list, dataset_score_list = [], []
    prev_dataset = None
    counter = 0
    dataset_name_patten = re.compile(r"(.*)-2020.*")
    for tup in score_list:
        counter += 1
        dataset_name = dataset_name_patten.findall(tup[0])[0]
        if dataset_name in ["coauthor", "az"]:
            dataset_name = "-".join(tup[0].split("-")[:2])
        if prev_dataset is not None and dataset_name != prev_dataset and len(dataset_score_list) > 0:
            mean_score_list.append(ScoreTuple(prev_dataset, dataset_score_list))
            dataset_score_list = []
        prev_dataset = dataset_name
        if tup[-1] is not None:
            dataset_score_list.append(float(tup[-1]))
    if len(dataset_score_list) > 0:
        mean_score_list.append(ScoreTuple(prev_dataset, dataset_score_list))

    mean_score_list = sorted(mean_score_list, key=lambda x: len(x.name))
    for tup in mean_score_list:
        print(tup)
        # print("{:<15} {}".format(tup[0], str(tup[1])))

if __name__ == '__main__':
    main(sys.argv)