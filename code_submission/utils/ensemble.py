import numpy as np


def get_top_models_by_std(models_info, ensemble_std_threshold=1e-2):
    """
    select model by std
    Args:
        models_info: (model, metric), where smaller metric indicates better model
        ensemble_std_threshold: std threshold
    Returns:

    """
    pred, metrics = zip(*sorted(models_info, key=lambda x: -x[1]))

    print("sorted model metrics:")
    top_num = 0
    for i in range(len(metrics)):
        print("metrics: {}".format(metrics[i]))
        std = np.std(metrics[:i])
        top_num = i
        if std > ensemble_std_threshold:
            break
    pred = pred[:top_num]
    metrics = np.asarray(metrics[:top_num])
    metrics = metrics + 15 * (metrics - metrics.mean())
    # metrics[np.where(metrics > 0.01)] = 0.01
    weights = metrics / metrics.sum()
    return list(zip(pred, weights))


def get_top_models_by_r(models_info, range_threshold=1e-2):
    """
    select model by std
    Args:
        models_info: (model, metric), where smaller metric indicates better model
        range_threshold: range threshold
    Returns:

    """
    pred, metrics, model_name = zip(*sorted(models_info, key=lambda x: -x[1]))

    print("sorted model metrics:")
    top_num = 0
    for i in range(len(metrics)):
        print("metrics: {}\tmodel_name: {}".format(metrics[i], model_name[i]))
        r = np.abs(metrics[0] - metrics[i])
        top_num = i
        if r > range_threshold:
            break
        if i == len(metrics)-1:
            top_num = i + 1
    pred = pred[:top_num]
    metrics = np.asarray(metrics[:top_num])
    metrics = metrics + 15 * (metrics - metrics.mean())
    weights = metrics / metrics.sum()
    return list(zip(pred, weights))
