import numpy as np

def compute_graded_metrics(y_true, y_pred):
    """
    Calculate Graded Precision, Graded Recall, and Graded F1 Score based on Sawhney et al. (2021).
    
    Args:
        y_true: true labels list or array (e.g., [0, 2, 3])
        y_pred: predicted labels list or array (e.g., [0, 1, 3])
        
    Returns:
        dict: containing 'graded_precision', 'graded_recall', 'graded_f1'
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_total = len(y_true)
    
    # Formula implementation 
    # FP: proportion of predicted values > true values (overestimation of risk)
    fp_ratio = np.sum(y_pred > y_true) / n_total
    
    # FN: proportion of predicted values < true values (underestimation of risk)
    fn_ratio = np.sum(y_pred < y_true) / n_total
    
    # Graded Precision & Recall (based on Sawhney et al.'s definition)
    # GP = 1 - FP ratio (represents the degree of "no overestimation" in predictions)
    graded_precision = 1.0 - fp_ratio
    
    # GR = 1 - FN ratio (represents the degree of "no underestimation/missing" in predictions)
    graded_recall = 1.0 - fn_ratio
    
    # Graded F1
    if (graded_precision + graded_recall) == 0:
        graded_f1 = 0.0
    else:
        graded_f1 = 2 * (graded_precision * graded_recall) / (graded_precision + graded_recall)
        
    return {
        "graded_precision": round(graded_precision, 4),
        "graded_recall": round(graded_recall, 4),
        "graded_f1": round(graded_f1, 4)
    }