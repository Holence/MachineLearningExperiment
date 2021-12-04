import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from scipy.spatial import ConvexHull

class KMeans():

    def __init__(self, k: int) -> None:
        self.k = k
        self.cluster_center_list = [] # 簇的中心点
        self.sample_list = None # 样本
        self.sample_cluster_index_list = None # 样本对应的簇的下标

    def fit(self, X: np.ndarray, max_iters: int = None, plt_process: bool = False):
        if max_iters!=None and max_iters<=0:
            raise ValueError("max_iters should larger than zero!")
        
        # 每次fit都会将X保存至sample_list
        self.sample_list=X
        
        # 初始化cluster_center
        self.cluster_center_list=[]
        for i in range(self.k):
            self.cluster_center_list.append([])
            for property_index in range(self.sample_list.shape[1]):
                column=self.sample_list[:,property_index]
                # 连续值
                if np.issubdtype(type(column[0]),np.number):
                    # 在范围内随机分配
                    self.cluster_center_list[i].append( np.random.uniform(np.min(column),np.max(column),1).item() )
                # 离散值
                else:
                    # 按数量排序后，从多到少分配
                    values,index=np.unique(column,return_index=True)
                    if len(values)-1<i:
                        self.cluster_center_list[i].append( [(column[index[len(values)-1]], 1)] )
                    else:
                        self.cluster_center_list[i].append( [(column[index[i]], 1)] )

        
        # 如果是plt_process且是二维特征值输入，则画图
        if plt_process and self.sample_list.shape[1]==2:
            colors=np.random.random((self.k,1,3))
            initial=np.array(self.cluster_center_list.copy())

        while True:
            # 计算每个sample所属的cluster，并记录下对应的下标，存储到sample_cluster_index_list中
            self.sample_cluster_index_list = []
            for sample in self.sample_list:
                # 按照中心点给样本分类
                self.sample_cluster_index_list.append(self.predict(sample))
            self.sample_cluster_index_list=np.array(self.sample_cluster_index_list)
            
            # 如果是plt_process且是二维特征值输入，则画图
            if plt_process and self.sample_list.shape[1]==2:
                plt.cla() # 清除原有图像
                plt.title("k-meas")
                for i in range(self.k):
                    selected = self.sample_list[self.sample_cluster_index_list==i]
                    plt.scatter(selected[:, 0], selected[:, 1], c=colors[i])
                    
                    # 画出每个簇的凸包
                    if(len(selected)>2):
                        hull = ConvexHull(selected).vertices.tolist()
                        hull.append(hull[0])
                        plt.plot(selected[hull, 0], selected[hull, 1], 'c--')
                
                # 初始的cluster_center特殊标注一下
                plt.scatter(initial[:, 0], initial[:, 1], c='#00FF00',edgecolors='#FF0000')
                plt.pause(0.2)
                plt.show()

            # 更新cluster_center
            old_center_list=self.cluster_center_list.copy()
            self.cluster_center_list=[]
            for i in range(self.k):
                self.cluster_center_list.append([])
                for property_index in range(self.sample_list.shape[1]):
                    column=self.sample_list[self.sample_cluster_index_list==i,property_index]
                    
                    if len(column)==0:
                        # print("哦我的上帝，有一簇的中心点太过偏远，以致于没有样本靠近它。\n\n重新随机生成初始点试试~")
                        self.cluster_center_list=old_center_list
                        return False
                    
                    # 连续值
                    if np.issubdtype(type(column[0]),np.number):
                        # 均值即是cluster_center的property_value
                        self.cluster_center_list[i].append( np.mean(column, axis=0) )
                    # 离散值
                    else:
                        # 这里统计cluster内的同种property下的所有property_value，为每一种property_value计算权重
                        # 将[(property_value1,0.2),(property_value2,0.3),(property_value3,0.5)]作为cluster_center的property_value
                        temp=[]
                        for property_value in np.unique(column):
                            temp.append( (property_value, len(column[column==property_value]) / len(column)) )
                        self.cluster_center_list[i].append(temp)
                            
            
            # 如果没有更新了，退出
            if self.cluster_center_list==old_center_list:
                break
            
            # 如果设定了迭代次数
            if max_iters!=None:
                max_iters-=1
                # 迭代次数到了，退出
                if max_iters==0:
                    break
        
        

    def predict(self, x: np.ndarray , retrun_index:bool = True):
        """预测单条数据（只做了连续值）

        Args:
            x (np.ndarray): 单条x数据
            retrun_index (bool, optional): 返回中心点下标，而不是中心点. Defaults to True.

        Returns:
            [type]: 返回中心点下标（或者将retrun_index设为False，即返回中心点）
        """
        
        min_distance=float('inf')
        min_cluster_index=None
        # 计算样本x与每一个cluster_center的distance，选出最小的一个作为归属
        for cluster_index in range(self.k):
            
            cluster_center=self.cluster_center_list[cluster_index]
            distance=0
            
            # 遍历每一种property，计算x与cluster_center的distance
            for property_index in range(len(x)):
                
                property_value=x[property_index]
                cluster_center_property_value=cluster_center[property_index]
                if np.issubdtype(type(property_value),np.number):
                    distance+=self.distance_PropertyValue(property_index,property_value,cluster_center_property_value)                
                else:
                    for thing in cluster_center_property_value:
                        value=thing[0]
                        weight=thing[1]
                        distance+=self.distance_PropertyValue(property_index,property_value,value)*weight

            if distance<min_distance:
                min_distance=distance
                min_cluster_index=cluster_index
        
        if retrun_index:
            return min_cluster_index
        else:
            return self.cluster_center_list[min_cluster_index]
    
    def distance_PropertyValue(self, property_index: int, Va, Vb) -> float:
        
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
                distance+=(muai/mua+mubi/mub)**2
        
        return distance

    def distance_Sample(self, Sa:np.ndarray, Sb: np.ndarray) -> float:
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
            distance+=self.distance_PropertyValue(i,Sa[i],Sb[i])
        
        return distance

    def score(self, method: Literal['DBI', 'DI']):
        def avg(cluster_index):
            selected=self.sample_list[self.sample_cluster_index_list==cluster_index]
            dist=0
            for i in range(len(selected)):
                sa=selected[i]
                for j in range(i+1,len(selected)):
                    sb=selected[j]
                    dist += self.distance_Sample(sa,sb)
            return 2/self.k/(self.k-1)*dist
        
        def diam(cluster_index):
            selected=self.sample_list[self.sample_cluster_index_list==cluster_index]
            max_dist=0
            for i in range(len(selected)):
                sa=selected[i]
                for j in range(i+1,len(selected)):
                    sb=selected[j]
                    dist=self.distance_Sample(sa,sb)
                    if dist>max_dist:
                        max_dist=dist
            return max_dist
        
        def d_min(cluster_index1, cluster_index2):
            selected1=self.sample_list[self.sample_cluster_index_list==cluster_index1]
            selected2=self.sample_list[self.sample_cluster_index_list==cluster_index2]
            min_dist=float('inf')
            for sa in selected1:
                for sb in selected2:
                    dist=self.distance_Sample(sa,sb)
                    if dist<min_dist:
                        min_dist=dist
            return min_dist
        
        def d_cen(cluster_index1, cluster_index2):
            center1=self.cluster_center_list[cluster_index1]
            center2=self.cluster_center_list[cluster_index2]
            if len(center1)!=len(center2):
                raise ValueError("len(center1)!=len(center2)!")
            
            dist=0
            for property_index in range(len(center1)):
                cluster_center_property_value1=center1[property_index]
                cluster_center_property_value2=center2[property_index]
                if np.issubdtype(type(cluster_center_property_value1),np.number):
                    dist+=self.distance_PropertyValue(property_index,cluster_center_property_value1,cluster_center_property_value2)                
                else:
                    for thing1 in cluster_center_property_value1:
                        value1=thing1[0]
                        weight1=thing1[1]
                        for thing2 in cluster_center_property_value2:
                            value2=thing2[0]
                            weight2=thing2[1]
                            dist+=self.distance_PropertyValue(property_index,value1,value2)*weight2*weight1
            return dist
        
        if method=='DBI':
            sum=0
            for i in range(self.k):
                max=0
                for j in range(self.k):
                    if i!=j:
                        res=(avg(i)+avg(j)) / (d_cen(i,j)+1e-16)
                        if res>max:
                            max=res
                sum+=max
            
            sum=sum/self.k

            return sum

        elif method=='DI':
            max_diam=0
            for i in range(self.k):
                res=diam(i)
                if res>max_diam:
                    max_diam=res
            
            min_d_min=float('inf')
            for i in range(self.k):
                for j in range(i+1,self.k):
                    res=d_min(i,j)
                    if res<min_d_min:
                        min_d_min=res
            
            return min_d_min/(max_diam+1e-16)
        
        else:
            raise ValueError("method should be in ['DBI', 'DI']")
