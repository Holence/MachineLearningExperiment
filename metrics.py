import numpy as np

def accuracy_score(Y_origin: np.ndarray, Y_predict: np.ndarray) -> float:
    if Y_origin.ndim!=1:
        raise ValueError("Y_origin.ndim!=1, which Y_origin.ndim == %s"%Y_origin.ndim)
    if Y_predict.ndim!=1:
        raise ValueError("Y_predict.ndim!=1, which Y_predict.ndim == %s"%Y_predict.ndim)
    if len(Y_origin)!=len(Y_predict):
        raise ValueError("len(Y_origin)!=len(Y_predict), which len(Y_origin) == %s and len(Y_predict) == %s"%(len(Y_origin),len(Y_predict)))
    
    return len(Y_predict[Y_predict==Y_origin]) / len(Y_origin)