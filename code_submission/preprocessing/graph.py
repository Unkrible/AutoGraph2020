import numpy as np
import pandas as pd


def is_undirected(num_node, edges):
    src, dist = zip(*edges)
    a_mat = np.zeros(shape=(num_node, num_node), dtype=np.bool)
    a_mat[src, dist] = True
    return (a_mat == a_mat.T).all()


def extend_directed(edges):
    """
    将无向图转为有向图
    Args:
        edges: pd.DataFrame(columns=['src_idx', 'dst_idx', 'edge_weight'])

    Returns:
        undirected_edges
    """
    edges_shadow = edges.copy()
    edges_shadow[['src_idx', 'dst_idx']] = edges_shadow[['dst_idx', 'src_idx']]
    undirected_edges = pd.concat([edges, edges_shadow], axis=0).reset_index(drop=True).drop_duplicates()
    return undirected_edges


def extract_graph_feature(graph_df, n_class):
    """

    Args:
        graph_df: {
                'fea_table': pd.DataFrame['node_index', 'feat_1', ..., 'feat_n'],
                'edge_file': pd.DataFrame['src_idx', 'dst_idx', 'edge_weight'],
                'train_indices': list of the index of train set,
                'test_indices': list of the index of test set,
                'train_label': pd.DataFrame['node_index', 'label']
            }
        n_class: num of class

    Returns:

    """
    fea_table = graph_df['fea_table'].set_index(keys="node_index")
    edges = graph_df['edge_file']
    train_indices = graph_df['train_indices']
    test_indices = graph_df['test_indices']
    train_label = graph_df['train_label']

    edge_weight = edges['edge_weight']
    in_degree = edges['dst_idx'].value_counts()
    out_degree = edges['src_idx'].value_counts()
    label_counts = train_label['label'].value_counts()

    (n_node, n_feature), n_edge = fea_table.shape, len(edges)
    n_train, n_test = len(train_indices), len(test_indices)
    meaning_weight = not (edge_weight == edge_weight[0]).all()
    max_degree, min_degree, mean_degree = in_degree.max(), in_degree.min(), in_degree.mean()
    max_labels, min_labels = label_counts.max(), label_counts.min()
    label_distribute = label_counts.sort_index(axis=0) / n_train
    print("label_distribute\n{}".format(label_distribute))
    info = {
        "n_node": n_node, "n_feature": n_feature, "n_edge": n_edge,
        "n_class": n_class,
        "n_train": n_train, "n_test": n_test,
        "meaning_weight": meaning_weight,
        "max_degree": max_degree, "min_degree": min_degree, "mean_degree": mean_degree,
        "max_labels": max_labels / n_train, "min_labels": min_labels / n_train,
        # "label_distribute": label_distribute
    }

    return info
