import numpy as np

def Gini(y: np.ndarray, weights: np.ndarray) -> float:
    "计算y集合的基尼值（带权重）"

    types_in_y = np.unique(y)
    
    gini = 1

    # 如果是普通样本
    if type(weights)==type(None):
        total=len(y)
        for one_type_in_y in types_in_y:
            pk = len(y[y == one_type_in_y]) / total
            gini -= pk ** 2
    
    # 如果是带weight的样本
    else:
        total_weight = np.sum(weights)
        for one_type_in_y in types_in_y:
            #这里的pk本来计算y值的概率（频度），因为要考虑weight，则直接统计 weight加和 占 总weight 的多少即可
            pk = np.sum(weights[y == one_type_in_y]) / total_weight
            gini -= pk ** 2

    return gini

def Continuous_Value_Gini_Index(column: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple:
    "计算被连续值属性划分后的y集合的最小基尼指数与划分点"
    sorted_x=np.sort(column)
    
    # 计算连续值的分裂点列表
    split_points = [(sorted_x[i] + sorted_x[i + 1]) / 2 for i in range(sorted_x.shape[0] - 1)]

    gini_min=float('inf')
    gini_min_split_point=None
    # 如果是普通样本
    if type(weights)==type(None):
        total=len(sorted_x)
        for i in split_points:
            y_left = y[column <= i]
            gini_left=Gini(y_left,weights) * len(y_left)/total

            y_right = y[column > i]
            gini_right=Gini(y_right,weights) * len(y_right)/total

            gini_whole=gini_left+gini_right

            if gini_whole<gini_min:
                gini_min=gini_whole
                gini_min_split_point=i
    
    # 如果是带weight的样本
    else:
        
        total_weight=np.sum(weights)
        for i in split_points:
            y_left = y[column <= i]
            weight_left = weights[column <= i]
            # 这里本来是加权加和各个y区的基尼值，因为要考虑weight，则计算各个y区的 weight加和 占 总weight 的多少即可
            gini_left=Gini(y_left,weight_left) * np.sum(weight_left)/total_weight

            y_right = y[column > i]
            weight_right = weights[column > i]
            # 这里本来是加权加和各个y区的基尼值，因为要考虑weight，则计算各个y区的 weight加和 占 总weight 的多少即可
            gini_right=Gini(y_right,weight_right) * np.sum(weight_right)/total_weight

            gini_whole = gini_left + gini_right

            if gini_whole<gini_min:
                gini_min=gini_whole
                gini_min_split_point=i

    return gini_min, gini_min_split_point

def Discrete_Value_Gini_Index(column: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    "计算被离散值属性划分后的y集合的基尼指数"

    split_points = np.unique(column)

    gini_min=float('inf')
    gini_min_split_point=None

    # 如果是普通样本
    if type(weights)==type(None):
        total=len(column)
        for one_type_in_column in split_points:

            y_selected = y[column == one_type_in_column]
            gini_selected = Gini(y_selected,weights) * len(y_selected)/total
            
            y_not_selected = y[column != one_type_in_column]
            gini_not_selected = Gini(y_not_selected,weights) * len(y_not_selected)/total
            
            gini_whole = gini_selected + gini_not_selected

            if gini_whole<gini_min:
                gini_min=gini_whole
                gini_min_split_point=one_type_in_column
    
    # 如果是带weight的样本
    else:
        total_weight=np.sum(weights)
        
        for one_type_in_column in split_points:
            # 二分类
            y_selected = y[column == one_type_in_column]
            weights_selected = weights[column == one_type_in_column]
            # 这里本来是加权加和各个y区的基尼值，因为要考虑weight，则计算各个y区的 weight加和 占 总weight 的多少即可
            gini_selected = Gini(y_selected,weights_selected) * np.sum(weights_selected)/total_weight
            
            # 二分类
            y_not_selected = y[column != one_type_in_column]
            weights_not_selected = weights[column != one_type_in_column]
            # 这里本来是加权加和各个y区的基尼值，因为要考虑weight，则计算各个y区的 weight加和 占 总weight 的多少即可
            gini_not_selected = Gini(y_not_selected,weights_not_selected) * np.sum(weights_not_selected)/total_weight

            gini_whole = gini_selected + gini_not_selected
            

            if gini_whole<gini_min:
                gini_min=gini_whole
                gini_min_split_point=one_type_in_column

    
    return gini_min, gini_min_split_point