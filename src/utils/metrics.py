from sklearn.metrics import f1_score

def compute_f1_score(y_true, y_pred):

    y_true = y_true.numpy().flatten()
    y_pred = y_pred.numpy().flatten()
    return f1_score(y_true, y_pred)