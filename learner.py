import numpy as np
from utils import Continuous_Value_Gini_Index, Discrete_Value_Gini_Index

# 通用学习器的基类
class Learner():
    def __init__(self, *args) -> None:
        pass

    def fit(self, X: np.ndarray, Y: np.ndarray, weight=None):
        pass

    def predict(self, x: np.ndarray):
        pass
    
    def predictBatch(self, X: np.ndarray) -> np.ndarray:
        res=[]
        for i in range(X.shape[0]):
            res.append(self.predict(X[i,:]))
        return np.array(res)

class NaiveBayesClassifier(Learner):
    def __init__(self, laplacian_correction=False, has_weight=False) -> None:
        self.laplacian_correction=laplacian_correction
        self.has_weight=has_weight
        self.p_dict={} # 概率字典
    
    def fit(self, X: np.ndarray, Y: np.ndarray, weight=None):
        
        # 不知道怎么贝叶斯怎么用weight
        # if self.has_weight:
        #     ma = weight >= np.mean(weight)*0.9
        #     X = X[ma]
        #     Y = Y[ma]

        num_Y=len(Y) # 样本总数
        
        Y_types=np.unique(Y) # y值的种类
        num_Y_types=len(Y_types) # y值种类的个数

        # 遍历每一种y值
        for y in Y_types:
            
            y_mask = Y==y
            Dy = Y[y_mask] # 单类y值对应的y们
            num_Dy = len(Dy) # 单类y值的个数
            
            # 计算该y值的先验概率
            if self.laplacian_correction:
                prior = (num_Dy+1) / (num_Y+num_Y_types)
            else:
                prior = num_Dy / num_Y

            # 后验概率字典
            posterior_dict={}
            # 这里直接用属性的下标(int)作为键了
            # 连续值的情况，值是一个tuple(mean,var)
            # 离散值的情况，值是一个float
            
            # 计算该y值下每一种属性的后验概率
            for property_index in range(X.shape[1]):
                Dp = X[:,property_index] # 单类y值对应的单种属性的属性值们（一列属性值）

                # 连续值
                if np.issubdtype(type(Dp[0]),np.number):
                    Dp=Dp[y_mask]
                    posterior_dict[property_index] = ( np.mean(Dp), np.sqrt(np.var(Dp)) )
                # 离散值
                else:
                    Dp_types = np.unique(Dp) # 单列属性的属性值的种类们
                    num_Dp_types = len(Dp_types) # 单列属性的属性值的种类数
                    posterior_dict[property_index]={}
                    for p_v in Dp_types:
                        p_v_mask = np.logical_and(y_mask,Dp==p_v) # 该y值下且属性值为p_v的
                        Dp_v = Dp[p_v_mask] # 属性值为p_v的一伙
                        num_Dp_v = len(Dp_v) # 属性值为p_v的数量
                        
                        if self.laplacian_correction:
                            posterior_dict[property_index][p_v] = (num_Dp_v+1) / (num_Dy+num_Dp_types)
                        else:
                            posterior_dict[property_index][p_v] = num_Dp_v / num_Dy
            
            self.p_dict[y]={
                "Prior": prior, # 先验概率
                "Posterior": posterior_dict # 后验概率字典
            }
        
    def predict(self, x: np.ndarray):
        
        # 概率列表
        PP_dict={}
        for key in self.p_dict.keys():
            PP_dict[key]=0

        # 遍历每一种y值，计算概率
        for key in self.p_dict.keys():
            
            # 先验概率
            prior=self.p_dict[key]["Prior"]
            PP_dict[key]+=np.log(prior)
            
            # 后验概率
            posterior_dict=self.p_dict[key]["Posterior"]
            for property_index,thing in enumerate(posterior_dict.values()):
                
                p_v=x[property_index] # x的属性值

                # 离散值，装有各种离散属性值的后验概率
                if type(thing)==dict:
                    PP_dict[key]+=np.log(thing[p_v])
                # 连续值，装有连续属性的平均值与标准差
                elif type(thing)==tuple:
                    mean,var=thing
                    PP_dict[key]+=np.log( 1/np.sqrt(2*np.pi)/ max(var,1e-16) *np.exp(-( (p_v-mean)**2) / max(2*var**2,1e-16) ) + 1e-16)
        
        # 按概率排序
        PP_dict=sorted(PP_dict.items(),key=lambda x:x[1],reverse=True)
        return PP_dict[0][0]
        

class DecisionTreeNode():
    def __init__(self) -> None:
        
        self.node_type=None # "root" or "leaf"
        
        # root
        self.left_tree=None # left tree node
        self.right_tree=None # right tree node

        # 属性在表格列中对应的 下标 和 属性种类
        # 色泽->(0, 青绿)
        # 如果是连续值的话，为 下标 和 划分点
        # 密度->(6, 0.25)
        self.spliting_property_index_and_types=None
        self.data_type="" # "Continuous" or "Discrete"
        
        # leaf
        self.result=None # None or y值

class DecisionTreeClassifier(Learner):
    def __init__(self, max_depth=4, has_weight=False) -> None:
        self.max_depth=max_depth # 树的最大深度（限制）
        self.has_weight=has_weight # 是否需要给样本赋权重，在adabooost中会用到
        self.depth=0 # 树的实际深度

    def fit(self, X: np.ndarray, Y: np.ndarray, weight=None) -> None:
        if self.has_weight and type(weight)==type(None):
            raise("has_weight设为True，意味着要传入参数的权重！")
        
        self.tree=self.creat_tree(X,Y,weight,0)
        # self.printTree()
    
    def creat_tree(self, X: np.ndarray, Y: np.ndarray, weight, depth) -> DecisionTreeNode:

        depth+=1

        # 记录树的深度
        if depth>self.depth:
            self.depth=depth
        
        current_node=DecisionTreeNode()
        
        # 只剩一个样本或者，所有的y都是同一类了，设置为叶节点
        if X.shape[0]==1 or len(np.unique(Y))==1:
            current_node.node_type="leaf"
            current_node.result=Y[0]
            return current_node
        
        # 限制深度，设置为叶节点
        if depth>=self.max_depth:
            current_node.node_type="leaf"
            
            # 深度受限，选择y中最多的一种作为最终的类别
            count={}
            for y_type in np.unique(Y):
                mask = Y==y_type
                if self.has_weight:
                    count[y_type] = np.sum(weight[mask])
                else:
                   count[y_type] = len(Y[mask])

            current_node.result=sorted(count.items(),key=lambda x :x[1],reverse=True)[0][0]

            return current_node
        
        # 继续深入建树

        # 按列遍历，计算每一列（每一个属性）的基尼指数，选出最小的一种属性
        gini_index_list=[] # 存储m个属性的(分类后的基尼指数，属性下标，连续|离散，划分点|属性类型)
        for property_index in range(X.shape[1]):
            column=X[:,property_index]
            
            if np.issubdtype(type(column[0]),np.number):
                TYPE="Continuous"
                gini_index,split_point=Continuous_Value_Gini_Index(column,Y,weight)
                
            else:
                TYPE="Discrete"
                gini_index,split_point=Discrete_Value_Gini_Index(column,Y,weight)
            
            gini_index_list.append( (gini_index, property_index, TYPE, split_point) )
        
        # 
        gini_index_list.sort(key=lambda x:x[0]) # 按照 基尼指数 升序排序，只取出 基尼指数 最小的用作当前层的划分
        
        # 选择最好的划分，深入建树
        BEST_ONE=gini_index_list[0] # 基尼指数最小的一种划分
        index=BEST_ONE[1]
        current_node.data_type=BEST_ONE[2]
        current_node.node_type="root"

        # 连续值，在划分点两侧分别建树
        if current_node.data_type=="Continuous":
            split_point=BEST_ONE[3]
            current_node.spliting_property_index_and_types=(index,split_point) # 连续值存储 属性下标 和 划分点

            left=X[:,index] <= split_point
            if len(X[left,:])!=0:
                if self.has_weight:
                    selected_weight=weight[left]
                else:
                    selected_weight=None
                
                current_node.left_tree=self.creat_tree( X[left,:], Y[left], selected_weight, depth)
            
            right=X[:,index] > split_point
            if len(X[right,:])!=0:
                if self.has_weight:
                    selected_weight=weight[right]
                else:
                    selected_weight=None
                
                current_node.right_tree=self.creat_tree( X[right,:], Y[right], selected_weight, depth)
        
        # 离散值，每种属性种类分别建树
        elif current_node.data_type=="Discrete":
            split_point=BEST_ONE[3]
            current_node.spliting_property_index_and_types=(index,split_point) # 离散值存储 属性下标 和 属性种类
            
            selected = X[:,index] == split_point
            if len(X[selected,:])!=0:
                if self.has_weight:
                    selected_weight=weight[selected]
                else:
                    selected_weight=None
                current_node.left_tree=self.creat_tree(X[selected,:], Y[selected], selected_weight, depth)
            
            others = X[:,index] != split_point
            if len(X[others,:])!=0:
                if self.has_weight:
                    others_weight=weight[others]
                else:
                    others_weight=None
                current_node.right_tree=self.creat_tree(X[others,:], Y[others], others_weight, depth)

        return current_node
    
    def predict(self, x: np.ndarray):
        
        def deepin(node:DecisionTreeNode):
            if node.node_type=="leaf":
                return node.result
            else:
                property_index,property_types=node.spliting_property_index_and_types
                property_given=x[property_index]
                
                if node.data_type=="Continuous":
                    if property_given<=property_types:
                        if node.left_tree!=None:
                            return deepin(node.left_tree)
                        # 可能左子树是空的
                        else:
                            return deepin(node.right_tree)
                    else:
                        if node.right_tree!=None:
                            return deepin(node.right_tree)
                        # 可能右子树是空的
                        else:
                            return deepin(node.left_tree)
                
                else:
                    if property_given==property_types:
                        if node.left_tree!=None:
                            return deepin(node.left_tree)
                        # 可能左子树是空的
                        else:
                            return deepin(node.right_tree)
                    else:
                        if node.right_tree!=None:
                            return deepin(node.right_tree)
                        # 可能右子树是空的
                        else:
                            return deepin(node.left_tree)

        return deepin(self.tree)
    
    def printTree(self):
        
        def deepin(node:DecisionTreeNode,depth):
            depth+=1
            if node.node_type=="root":
                print("\t"*depth,node.node_type,node.spliting_property_index_and_types)
                if node.left_tree!=None:
                    deepin(node.left_tree,depth)
                if node.right_tree!=None:
                    deepin(node.right_tree,depth)
            else:
                print("\t"*depth,node.node_type,node.result)
        
        depth=-1
        deepin(self.tree,depth)

class KNN(Learner):
    def __init__(self, k) -> None:
        self.k=k
    
    def fit(self, X: np.ndarray, Y: np.ndarray, weight=None):
        self.sample_list=X
        self.tag_list=Y
    
    def predict(self, x: np.ndarray):

        dist_list=[]
        for sample,tag in zip(self.sample_list,self.tag_list):
            dist=self._distance_Sample_Sample(sample,x)
            dist_list.append((dist,tag))
        dist_list.sort(key=lambda x:x[0])
        
        tag_dict={}
        for i in range(self.k):
            tag=dist_list[i][1]
            
            if tag_dict.get(tag)==None:
                tag_dict[tag]=1
            else:
                tag_dict[tag]+=1
        
        return sorted(tag_dict.items(),key=lambda x:x[1],reverse=True)[0][0]
    
    def _distance_Value_Value(self, property_index: int, Va, Vb) -> float:
        """属性值与属性值的距离（可以是连续值也可以是离散值）

        Args:
            property_index (int): 属性列的下标
            Va ([type]): 属性值Va
            Vb ([type]): 属性值Vb

        Returns:
            float: 属性值与属性值的距离
        """        
        
        # 连续值
        if np.issubdtype(type(Va),np.number) and np.issubdtype(type(Vb),np.number):
            distance=(Va-Vb)**2
        # 离散值
        else:
            # VDM
            column=self.sample_list[:,property_index]
            mua=len(column[column==Va])+1e-16
            mub=len(column[column==Vb])+1e-16
            distance=0
            for i in range(self.k):
                muai=len(column[ np.logical_and(column==Va, self.sample_cluster_index_list==i) ])
                mubi=len(column[ np.logical_and(column==Vb, self.sample_cluster_index_list==i) ])
                distance+=(muai/mua-mubi/mub)**2
        
        return distance**0.5
    
    def _distance_Sample_Sample(self, Sa:np.ndarray, Sb: np.ndarray) -> float:
        """样本与样本的距离

        Args:
            Sa (np.ndarray): 样本Sa
            Sb (np.ndarray): 样本Sb

        Returns:
            float: 样本与样本的距离
        """        
        if Sa.ndim!=1:
            raise ValueError("Sa.ndim!=1, where Sa.ndim == %s"%Sa.ndim)
        if Sb.ndim!=1:
            raise ValueError("Sb.ndim!=1, where Sb.ndim == %s"%Sb.ndim)
        if len(Sa)!=self.sample_list.shape[1]:
            raise ValueError("len(Sa)!=self.sample_list.shape[1], where len(Sa) == %s"%len(Sa))
        if len(Sb)!=self.sample_list.shape[1]:
            raise ValueError("len(Sb)!=self.sample_list.shape[1], where len(Sb) == %s"%len(Sb))

        distance=0
        for i in range(len(Sa)):
            distance+=self._distance_Value_Value(i,Sa[i],Sb[i])
        
        return distance