import numpy as np

def business_cost(y_true, y_pred, cost_fn=10, cost_fp=1):
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return cost_fn * fn + cost_fp * fp

def test_business_cost_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])  # 1 FP + 1 FN
    assert business_cost(y_true, y_pred, cost_fn=10, cost_fp=1) == 11
