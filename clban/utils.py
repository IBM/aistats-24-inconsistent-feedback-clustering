import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment


def align_clusters(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Aligns the clusters using the Hungarian algorithm for matching cluster indices by using pairwise F1 score between
    clusters as a criterion

    :param y_true: A (num_arms,) array containing ground truth clusters
    :param y_pred: A (num_arms,) array containing estimated clusters
    :return y_aligned: An aligned version of y_pred
    """
    # Calculate the cost metric
    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)
    costs = np.zeros(((np.max(y_pred) + 1), np.max(y_true) + 1))
    for idx1 in range(costs.shape[0]):
        for idx2 in range(costs.shape[1]):
            costs[idx1, idx2] = 1 - f1_score(y_true == idx2, y_pred == idx1, zero_division=0.0)

    # Find cluster assignment
    row_index, col_index = linear_sum_assignment(costs)
    cluster_map = dict((row_index[idx], col_index[idx]) for idx in range(row_index.shape[0]))
    for idx in range(y_pred.shape[0]):
        if y_pred[idx] not in cluster_map:
            cluster_map[y_pred[idx]] = y_pred[idx]

    return np.asarray([cluster_map[y_pred[idx]] for idx in range(y_pred.shape[0])])
