"""the simple baseline for autograph"""
import random

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.utils as gtils
from collections import defaultdict
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from scipy.stats import gmean

from models import *
from models import MODEL_PARAMETER_LIB
from utils import *
from preprocessing import *
from config import Config
from utils.ensemble import get_top_models_by_std, get_top_models_by_r
from utils.drop_edge import DropEdgeEachStep

import copy
import gc


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


logger = get_logger("INFO", use_error_log=True)


class Model:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = None
        self.metadata = {}
        self._num_nodes = None
        self._origin_graph_data_indices = None
        self._valid_indices = None
        self._valid_mask = None
        self._train_indices = None
        self._train_mask = None
        self._test_mask = None
        self._sampler = None
        self._n_class = None
        self.y_train = None
        self.models_topK = defaultdict(list)
        self.used_model_num = 0
        self.citation_configs = ['a', 'b', 'demo', 'coauthor-cs', 'coauthor-phy', 'phy10000']
        self.use_adaptive_topK = True

    def load_config(self, data, n_class):
        dir_path = os.path.dirname(__file__)
        try:
            tree = joblib.load(f"{dir_path}/meta.model")
            encoder = joblib.load(f"{dir_path}/meta.encoder")
            # pd.set_option('display.max_columns', None)
            meta_info = pd.Series(
                extract_graph_feature(data, n_class)
            )
            logger.info("meta_info:\n {}".format(meta_info))
            meta_info = pd.DataFrame([meta_info])
            self.metadata = meta_info

            logger.error(f"tree prob:{tree.predict_proba(meta_info)}")
            if meta_info['n_feature'][0] == 0:
                logger.error("n_feature of this set is 0")
                config = "e"
            else:
                config = encoder.inverse_transform(tree.predict(meta_info))[0]
            if config == "e" and meta_info['n_class'].iloc[0] >= 5:
                config = "e-d5"
            logger.error(f"use {config} config by meta learning")
            self.config = Config(f"{dir_path}/configs/{config}.json")
            # self.config = Config(f"{dir_path}/configs/tmp.json")
        except Exception as e:
            logger.error("Throw error when loading config")
            logger.error(e)
            self.config = Config(f"{dir_path}/configs/default.json")
            # self.config = Config(f"{dir_path}/configs/tmp.json")

    def train_valid_split(self, total_indices, y, valid_rate=0.2):
        total_indices = np.asarray(total_indices, dtype=np.int32)
        total_class_indices = []
        train_indices, valid_indices = [], []
        each_class_max_sample_num = 1000
        for i in range(self._n_class):
            total_class_indices.append(np.where(y[:] == i)[0])
            each_class_valid_num = max(1, int(len(total_class_indices[i])*valid_rate))
            each_class_valid_indices = np.random.choice(total_class_indices[i],
                                                        each_class_valid_num,
                                                        replace=False).tolist()
            each_class_train_indices = list(set(total_class_indices[i]) - set(each_class_valid_indices))
            if len(each_class_train_indices) == 0:
                each_class_train_indices = each_class_valid_indices
            train_indices += np.random.permutation(each_class_train_indices)[:each_class_max_sample_num].tolist()
            valid_indices += each_class_valid_indices

        train_indices, valid_indices = total_indices[train_indices], total_indices[valid_indices]
        random.shuffle(train_indices)
        random.shuffle(valid_indices)
        return train_indices, valid_indices

    def generate_pyg_data(self, data):
        # get x feature table
        x = data['fea_table'].copy()
        df = data['edge_file']
        edges = df[['src_idx', 'dst_idx', 'edge_weight']]

        # get indices first
        train_indices = data['train_indices']
        if self.config.use_valid:
            train_indices, valid_indices = train_test_split(train_indices, test_size=0.2, shuffle=False)

        try:
            if x.shape[1] == 1:        # 0-dimensional feature
                x = x.set_index(keys="node_index")
                x = feat_engineering(
                    x,
                    edges=edges,
                    num_nodes=self.metadata["n_node"].iloc[0]
                )
            else:
                x_feat = x.drop('node_index', axis=1).to_numpy()
                conf_name = self.config.filename.split("/")[-1].split(".")[0]
                is_only_one_zero = not ((x_feat != 0) & (x_feat != 1)).any()
                logger.info("use {} config".format(conf_name))
                logger.info(
                    "feature only contains zero: {}, only one and zero: {}".format((x_feat == 0).all(), is_only_one_zero))

                if conf_name in self.citation_configs:  # Judge whether it is a citation graph
            # if True:
                    if is_only_one_zero:
                        logger.info("Normalize features")
                        normal_feat = feat_row_sum_inv_normalize(x_feat)
                        normal_df = pd.DataFrame(data=normal_feat)
                        normal_df["node_index"] = x["node_index"]
                        x = normal_df

                    pre_feat = prepredict(data, train_indices=train_indices, use_valid=self.config.use_valid, use_ohe=False)
                    x = x.set_index(keys="node_index")
                    x_index = x.index.tolist()
                    lpa_preds, lpa_train_acc = lpa_predict(data, n_class=self._n_class, train_indices=train_indices, use_valid=self.config.use_valid)
                    if not np.isnan(lpa_train_acc) and lpa_train_acc > 0.8:
                        logger.info("Use LPA predicts")
                        x = pd.concat([x, pre_feat, lpa_preds], axis=1).values[x_index]
                    else:
                        x = pd.concat([x, pre_feat], axis=1).values[x_index]
                else:
                    x = x.set_index(keys="node_index")
                    x = feat_engineering(
                        x,
                        edges=edges,
                        num_nodes=self.metadata["n_node"].iloc[0]
                    )
        except Exception as e:
            logger.error(e)
            if x.shape[1] == 0:
                x = np.zeros((x.shape[0], 64), dtype=np.float)
            else:
                x = x.to_numpy()

        logger.info("x shape: {}".format(x.shape))
        node_index = torch.tensor(data['fea_table']['node_index'].to_numpy(), dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)

        # get edge_index, edge_weight
        edges = edges.to_numpy()
        edge_index = edges[:, :2].astype(np.int)
        # transpose from [edge_num, 2] to [2, edge_num] which is required by PyG
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
        edge_weight = edges[:, 2]
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        undirected = gtils.is_undirected(edge_index)

        edge_index, edge_weight = gtils.sort_edge_index(edge_index, edge_weight)
        logger.info(f"is undirected ? {undirected}")
        logger.info(f"edge index {edge_index.shape}, edge weight {edge_weight.shape}")

        # get train/test mask
        num_nodes = x.size(0)
        self._num_nodes = num_nodes
        y = torch.zeros(num_nodes, dtype=torch.long)
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        self.y_train = train_y
        y[inds] = torch.tensor(train_y, dtype=torch.long)

        # train_indices = data['train_indices']
        self._origin_graph_data_indices = copy.deepcopy(data['train_indices'])
        if self.config.use_valid:
            # train_indices, valid_indices = train_test_split(train_indices, test_size=0.2)
            # train_indices, valid_indices = train_test_split(train_indices, test_size=0.2, shuffle=False)
            self.y_train = data['train_label'].set_index('node_index').loc[train_indices][['label']].to_numpy()
        test_indices = data['test_indices']

        data = Data(x=x, node_index=node_index, edge_index=edge_index, y=y, edge_weight=edge_weight)

        data.num_nodes = num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = 1
        data.train_indices = np.asarray(train_indices)
        data.train_mask = train_mask
        self._train_indices = np.asarray(train_indices)
        self._train_mask = train_mask

        if self.config.use_valid:
            valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
            valid_mask[valid_indices] = 1
            data.valid_indices = valid_indices
            data.valid_mask = valid_mask
            self._valid_indices = valid_indices
            self._valid_mask = valid_mask

        self._test_mask = np.zeros(num_nodes, dtype=np.bool)
        self._test_mask[test_indices] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_indices] = 1
        data.test_mask = test_mask
        data.test_indices = np.asarray(test_indices)

        self._sampler = Sampler(data, self.metadata["n_edge"].iloc[0], self.device)

        return data

    def train(self, sampler, n_class):

        try:
            time_budget = get_time_budget().timing(frac=0.95)
            drop_edge_controller = None
            model_time_budget = max(time_budget.remain * 0.6, time_budget.remain / len(self.config.model_classes))
            self.models_topK = defaultdict(list)

            for model_name, model_class in self.config.model_classes:
                time_budget.check()
                config = self.config.model_config(model_name)
                logger.info(f"model {model_name} config:\n{config}")
                self.used_model_num += 1
                data = sampler.random_edge_sampler(percent=config.drop_edge)

                model = model_class(
                    features_num=data.x.size()[1],
                    num_class=n_class,
                    edge_num=data.edge_index.shape[1],
                    num_layers=config.num_layers,
                    hidden=config.hidden,
                    dropout=config.dropout,
                    drop_edge_controller=drop_edge_controller,
                    num_nodes=self._num_nodes,
                    emb_dim=config.emb_dim
                )

                model = model.to(self.device)

                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                             weight_decay=config.weight_decay)

                train_kwargs = {}
                if config.use_valid:
                    train_kwargs = {
                        "valid_indices": data.valid_indices,
                    }

                if config.use_sampler:
                    dataset = GraphSampleDataset(data, n_class, self.y_train)
                else:
                    dataset = GraphDataset(data)

                topK_list, valid_acc, except_info = torch_train(
                    data, dataset, model, optimizer, F.nll_loss,
                    epochs=config.num_epoch, batch_size=data.num_nodes // config.num_batch,
                    min_epochs=config.min_epoch,
                    clip_grad=5 if config.use_sampler else 0,
                    patience=config.patience,
                    time_budget=time_budget,
                    all_data=False,
                    use_adaptive_topK=self.use_adaptive_topK,
                    model_topK=self.models_topK[model_name],
                    **train_kwargs
                )

                if except_info == "time_exceed":
                    print("execute to {}".format(except_info))
                    if -valid_acc < 0.80 and self.used_model_num > 1:
                        del model
                        gc.collect()
                        break

                    self.models_topK[model_name] = topK_list
                    del model
                    gc.collect()
                    break

                if except_info == "oom":
                    del model
                    gc.collect()
                    continue

                self.models_topK[model_name] = topK_list

                del model
                gc.collect()
                self.config.model_classes.update(self.models_info)
            return self.models_info

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("we met cuda out of memory")
                return self.models_info
            else:
                raise exception
        except TimeOutError as e:
            print(e)
            return self.models_info

    @property
    def models_info(self):
        info = []
        for name in self.models_topK:
            info.extend([(ele['pred'], ele['acc'], name) for ele in self.models_topK[name]])
        return info

    def transition_train_valid(self, data, target):
        if target == 'valid':
            data.valid_indices = self._valid_indices
            data.valid_mask = self._valid_mask
            data.train_indices = self._train_indices
            data.train_mask = self._train_mask
        else:
            train_mask = torch.zeros(self._num_nodes, dtype=torch.bool)
            train_indices = self._origin_graph_data_indices
            train_mask[train_indices] = 1
            data.train_indices = np.asarray(train_indices)
            data.train_mask = train_mask
        return data

    def fake_pred(self, model, data):
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            logits, labels = model(data.test_mask).max(1)
        return logits, labels

    def pred(self, model, data):
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            logits = model(data)
            _, preds = logits[data.test_mask].max(1)
        return logits, preds

    def train_predict(self, data, time_budget, n_class, schema):
        """
        API for ingestion prog to invoke
        Args:
            data: {
                'fea_table': pd.DataFrame['node_index', 'feat_1', ..., 'feat_n'],
                'edge_file': pd.DataFrame['src_idx', 'dst_idx', 'edge_weight'],
                'train_indices': list of the index of train set,
                'test_indices': list of the index of test set,
                'train_label': pd.DataFrame['node_index', 'label']
            }
            time_budget: remain time
            n_class: class num
            schema: deprecated

        Returns: prediction of nodes in test set

        """
        set_time_budget(time_budget)
        self._n_class = n_class
        self.load_config(data, n_class)
        data = self.generate_pyg_data(data)
        # model = self.train(data, n_class)
        models_info = self.train(self._sampler, n_class)
        print("models_info_acc:")
        for i in range(len(models_info)):
            print("acc: {}".format(models_info[i][1]))

        # test_logits, test_labels = self.fake_pred(model, data)
        # test_logits = test_logits.cpu().numpy()
        # test_sorted = test_logits.argsort()[::-1]
        # selected = test_sorted[: int(len(test_sorted) * 0.4)]
        # selected_id = data.test_indices[selected]
        # data.y[selected_id] = torch.tensor(test_labels.cpu().numpy().flatten()[selected], dtype=torch.long, device=self.device)
        # data.train_mask[selected_id] = 1
        # model = self.train(data)
        # logits, preds = self.pred(model, data)

        timing = get_time_budget().timing(frac=0.97)

        ensemble_info = get_top_models_by_r(models_info)

        try:
            logger.info("logits_ensemble_len: {}".format(len(ensemble_info)))

            logits_ensemble = None
            # logits_list = []
            for pred, weight in ensemble_info:
                timing.check()
                # logger.info("model_ensemble_weight: {}".format(weight))
                logits = pred[self._test_mask, :]
                # normalize logits
                logits = logits.T
                logits = (logits - np.min(logits, axis=0)) / (np.max(logits, axis=0) - np.min(logits, axis=0))
                logits = logits.T
                # logits_list.append(logits)
                if logits_ensemble is None:
                    logits_ensemble = logits * weight
                else:
                    logits_ensemble += logits * weight
                timing.check()

            # logits_ensemble = np.array(logits_list)
            # logger.info("use gmeans ensemble; logits_ensemble_shape: {}".format(logits_ensemble.shape))
            # logits_ensemble = gmean(logits_ensemble, axis=0)
            preds = np.argmax(logits_ensemble, axis=1)

            return preds.flatten()
        except TimeOutError as e:
            print(e)
            return np.argmax(ensemble_info[0][0][self._test_mask, :], axis=1).flatten()
        except Exception as e:
            print(e)
            return np.argmax(np.random.rand(self.metadata['n_test'].iloc[0], self._n_class), axis=1).flatten()

