import json
import numpy as np
from collections import defaultdict, ChainMap
from models import MODEL_LIB, MODEL_PARAMETER_LIB


class ModelList:
    def __init__(self, names, loop=False):
        self.names = names
        self.n_models = len(names)
        self.index = 0
        self.model_info = None
        self.loop = loop

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if not self.loop and self.index >= self.n_models:
            raise StopIteration
        self.index = self.index % self.n_models
        if self.index == 0 and self.model_info is not None:
            self._update()
        name = self.names[self.index]
        self.index = self.index + 1
        return name, MODEL_LIB[name]

    def __len__(self):
        return len(self.names)

    def _update(self):
        model_info = [(ele[1], ele[2]) for ele in self.model_info]
        model_metrics = defaultdict(list)
        for metric, name in model_info:
            model_metrics[name].append(metric)
        model_metrics = {name: np.mean(metrics) for name, metrics in model_metrics.items()}
        model_metrics = [(name, model_metrics.get(name, 1.0))for name in self.names]
        model_metrics = sorted(model_metrics, key=lambda x: x[1], reverse=True)
        print("sorted metrics", model_metrics)
        self.names = [ele[0] for ele in model_metrics]

    def update(self, model_info):
        self.model_info = model_info


class Config:
    """
    统一管理全局超参数, 如模型序列, 数据处理方式, batch size等
    """
    def __init__(self, filename="", config=None):
        self.filename = filename
        if config is None:
            self.config = defaultdict(lambda: None)
            with open(filename, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config
        self.model_list = None

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __delitem__(self, key):
        del self.config[key]

    def __getattr__(self, item):
        return self.config.get(item, None)

    def __str__(self):
        return str(self.config)

    @property
    def loop(self):
        return self.config.get('loop', False)

    @property
    def model_classes(self):
        if self.model_list is None:
            self.model_list = ModelList(self.model, self.loop)
        return self.model_list

    def model_config(self, name):
        config = {}
        if "model_config" in self.config:
            if name in self.config["model_config"]:
                config = self.config["model_config"][name]
            if name in MODEL_PARAMETER_LIB:
                params = MODEL_PARAMETER_LIB[name]
                config = ChainMap(config, {"lr": params[0], "dropout": params[1], "weight_decay": params[2], "hidden": params[3]})
        config = ChainMap(config, self.config)
        return Config(config=config)
