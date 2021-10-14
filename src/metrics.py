import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, jaccard_score
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, matthews_corrcoef
from sklearn.metrics import average_precision_score, roc_auc_score
## regression metrics
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score

def auprc(y_gt, prob, pi=1, average='binary', **kwargs):
    # determine if binary or multiclass
    if average == 'binary':
        pre_vals, rec_vals, _ = precision_recall_curve(y_gt, prob[:,pi])
        auprc = auc(rec_vals, pre_vals)
    else:
        nc = prob.shape[1]
        auprc = average_precision_score(np.eye(nc)[y_gt], prob, average=average, **kwargs)
        
    return auprc

def auroc(y_gt, prob, pi=1, average='binary', **kwargs):
    # determine if binary or multiclass
    # print(y_gt, prob)
    if average == 'binary':
        fpr, tpr, _ = roc_curve(y_gt, prob[:,pi])
        auroc = auc(fpr, tpr)
    else:
        nc = prob.shape[1]
        auroc = roc_auc_score(np.eye(nc)[y_gt], prob, average=average, **kwargs)
    
    return auroc
