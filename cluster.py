import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import random
from scipy.spatial import ConvexHull
from utils import logistic

class KMeans():

    def __init__(self, k: int = 2) -> None:
        self.k = k
        self.cluster_center_list = [] # 簇的中心点
        self.sample_list = None # 样本
        self.sample_cluster_index_list = None # 样本对应的簇的下标

    def fit(self, X: np.ndarray, max_iters: int = None, plt_process: bool = False, auto_k: bool = True):
        if max_iters!=None and max_iters<=0:
            raise ValueError("max_iters should larger than zero!")

        # 每次fit都会将X保存至sample_list
        self.sample_list=X
        
        # 选定初始点，先随机选一个样本，再生成k-1个
        # 选择的条件是与initial_selected中已选点的距离和最大 且 与每个已选点的距离不小于0.9*avg_dist
        initial_selected=[random.randint(0,self.sample_list.shape[0]-1)]
        avg_dist=self.__dist_avg(self.sample_list)
        for i in range(self.k-1):
            max_dist=0
            max_dist_index=None
            for Sb_index in range(self.sample_list.shape[0]):
                if Sb_index not in initial_selected:
                    Sb=self.sample_list[Sb_index]
                    dist=0
                    flag=True
                    for Sa_index in initial_selected:
                        Sa=self.sample_list[Sa_index]
                        res=self.__distance_Sample_Sample(Sa,Sb)
                        if res<avg_dist*0.9:
                            flag=False
                            break
                        else:
                            dist+=res
                    
                    if flag==True and dist>max_dist:
                        max_dist=dist
                        max_dist_index=Sb_index
            if max_dist_index==None:
                max_dist_index=initial_selected[0]
                while max_dist_index in initial_selected:
                    max_dist_index=random.randint(0,self.sample_list.shape[0]-1)
            initial_selected.append(max_dist_index)
        
        # 用initial_selected初始化self.cluster_center
        initial_selected=self.sample_list[initial_selected]
        self.cluster_center_list=[]  
        for i in initial_selected:
            cluster=np.array([i])
            self.cluster_center_list.append(self.__generate_Center(cluster))


        # 如果是plt_process且是二维特征值输入，则画图
        if plt_process and self.sample_list.shape[1]==2:
            # 一次fit一种配色
            self.colors=[]
            for i in range(self.k):
                self.colors.append(np.random.random((1,1,3)))
            self.initial_copied=np.array(self.cluster_center_list.copy())

        if auto_k:
            self.splitted_cluster=[]
        
        # 根据初始中心划分样本，之后便
        self.sample_cluster_index_list = []
        for sample in self.sample_list:
            # 按照中心点给样本分类
            self.sample_cluster_index_list.append(self.predict(sample))
        self.sample_cluster_index_list=np.array(self.sample_cluster_index_list)
        
        # 开始迭代
        while True:
            
            if not auto_k:
                # 计算每个sample所属的cluster，并记录下对应的下标，存储到sample_cluster_index_list中
                self.sample_cluster_index_list = []
                for sample in self.sample_list:
                    # 按照中心点给样本分类
                    self.sample_cluster_index_list.append(self.predict(sample))
                self.sample_cluster_index_list=np.array(self.sample_cluster_index_list)
            
            # 如果是plt_process且是二维特征值输入，则画图
            if plt_process and self.sample_list.shape[1]==2:
                self.plot()

            # 更新cluster_center
            old_center_list=self.cluster_center_list.copy()
            self.cluster_center_list=[]
            for i in range(self.k):
                cluster=self.sample_list[self.sample_cluster_index_list==i]
                self.cluster_center_list.append(self.__generate_Center(cluster))
            
            # 根据样本分布，自动增减k值
            if auto_k:

                # 簇太多了吗？
                i=0
                while i<self.k and self.k>2:
                    # 以i簇为起点，向其他簇试探
                    cluster1 = self.sample_list[self.sample_cluster_index_list==i]
                    neighbor1=self.__dist_max_neighbor(cluster1)
                    
                    j=i+1
                    while j<self.k and self.k>2:
                        cluster2 = self.sample_list[self.sample_cluster_index_list==j]
                        neighbor2=self.__dist_max_neighbor(cluster2)
                        edge=self.__dist_edge(cluster1,cluster2)

                        # 如果 i簇与j簇的 边缘距离 小于 各自簇内的邻居距离，则合并
                        if edge<=neighbor1*1.2 or edge<=neighbor2*1.2:

                            # cluster2并入cluster1
                            self.sample_cluster_index_list[self.sample_cluster_index_list==j] = i
                            
                            if i in self.splitted_cluster:
                                self.splitted_cluster.append(i)

                            # 删除cluster2
                            self.cluster_center_list.pop(j)
                            
                            if j in self.splitted_cluster:
                                self.splitted_cluster=[t for t in self.splitted_cluster if t!=j]

                            # 更新i簇的中心点
                            self.cluster_center_list[i]=self.__generate_Center(self.sample_list[self.sample_cluster_index_list==i])

                            # j之后的簇的下标向前移一位
                            o=j
                            while o<self.k-1:
                                self.sample_cluster_index_list[self.sample_cluster_index_list == o+1] = o
                                o+=1
                            
                            self.k-=1
                            j-=1

                            if plt_process and self.sample_list.shape[1]==2:
                                # 颜色的个数也要重新调整
                                self.colors.pop(j)
                                self.plot()
                        
                        j+=1
                
                    i+=1
                

                # 簇太少了吗？
                i=0
                one_more_try=False
                while i<self.k:
                    if plt_process and self.sample_list.shape[1]==2:
                        self.plot()
                    
                    
                    if self.splitted_cluster.count(i)>=2:
                        i+=1
                        continue

                    cluster=self.sample_list[self.sample_cluster_index_list==i]

                    origin_index_list=[]
                    o=0
                    for j in self.sample_cluster_index_list==i:
                        if j==True:
                            origin_index_list.append( o )
                        o+=1
                    
                    
                    full_neighbor=self.__dist_max_neighbor(cluster)
                    
                    picked=random.randint(0, len(cluster)-1)
                    checking_list=[picked]
                    done_list=[]
                    left_list=list(range(len(cluster)))
                    left_list.remove(picked)

                    # 以checking为起点，向周围试探
                    while checking_list!=[]:

                        new_checking_list=[]
                        for checking in checking_list:
                            
                            if left_list==[]:
                                break
                            
                            done_list.append(checking)
                            
                            # 最一开始threshold给较大的步幅
                            if len(done_list)<=2:
                                threshold=full_neighbor
                            # 等取到一定样本个数后，threshold变成__dist_max_neighbor
                            else:
                                cc=self.sample_list[[origin_index_list[done] for done in done_list]]
                                neighbor=self.__dist_max_neighbor(cc)
                                threshold = neighbor
                            
                            adding=[ left for left in left_list if self.__distance_Sample_Sample(cluster[checking],cluster[left]) < threshold]
                            
                            # 如果步幅仍太小，有一次提升步幅的机会
                            if adding==[]:
                                cc=self.sample_list[[origin_index_list[done] for done in done_list]]
                                var = 1 + logistic(self.__dist_var(cc)**2)
                                # avg=1+logistic(self.__dist_avg(cc))
                                # threshold = neighbor + avg*var
                                threshold *= var
                                adding=[ left for left in left_list if self.__distance_Sample_Sample(cluster[checking],cluster[left]) < threshold]
                            
                            # 以threshold为界限
                            new_checking_list += adding
                            left_list=[ left for left in left_list if left not in adding]
                        
                        if plt_process and self.sample_list.shape[1]==2:
                            having=[]
                            having+=done_list
                            having+=new_checking_list
                            cc=self.sample_list[[origin_index_list[h] for h in having]]
                            plt.scatter(cc[:, 0], cc[:, 1], c='#00FFFF',edgecolors='#5500FF',linewidths=2)
                            plt.pause(0.01)
                            
                        checking_list=new_checking_list
                    

                    # 如果最后还有剩下的，划分为新的一簇
                    if left_list!=[]:
                        
                        # 检验 边缘距离 与 各自簇内的邻居距离
                        done_list+=checking_list
                        done_list=[origin_index_list[done] for done in done_list]
                        dd=self.sample_list[done_list]

                        left_list=[origin_index_list[left] for left in left_list]
                        ll=self.sample_list[left_list]
                        
                        edge=self.__dist_edge(dd,ll)
                        neighbor_ll=self.__dist_max_neighbor(ll)
                        neighbor_dd=self.__dist_max_neighbor(dd)
                        if edge>neighbor_ll and edge>neighbor_dd:

                            # 给left_list中分配新的cluster_index
                            self.sample_cluster_index_list[left_list]=self.k
                            # 生成新的cluster_center
                            cluster=self.sample_list[self.sample_cluster_index_list==self.k]
                            self.cluster_center_list.append(self.__generate_Center(cluster))
                            
                            # 重新计算i簇的中心点
                            self.cluster_center_list[i]=self.__generate_Center(self.sample_list[self.sample_cluster_index_list==i])

                            self.splitted_cluster.append(i)

                            self.k+=1
                            if plt_process and self.sample_list.shape[1]==2:
                                # 颜色的个数也要重新调整
                                self.colors.append(np.random.random((1,1,3)))
                            
                            # 一次优化就分裂一次好了
                            # break
                        elif one_more_try==False:
                            one_more_try=True
                        else:
                            one_more_try=False
                    else:
                        self.splitted_cluster.append(i)
                        if one_more_try==False:
                            one_more_try=True
                        else:
                            one_more_try=False

                    if one_more_try!=True:
                        i+=1

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
            
            distance=self.__distance_Sample_Center(x,self.cluster_center_list[cluster_index])

            if distance<min_distance:
                min_distance=distance
                min_cluster_index=cluster_index
        
        if retrun_index:
            return min_cluster_index
        else:
            return self.cluster_center_list[min_cluster_index]
    
    def plot(self):
        plt.cla() # 清除原有图像
        plt.title("k-meas %d"%self.k)
        for i in range(self.k):
            cluster = self.sample_list[self.sample_cluster_index_list==i]
            plt.scatter(cluster[:, 0], cluster[:, 1], c=self.colors[i])
            
            # 画出每个簇的凸包
            try:
                hull = ConvexHull(cluster).vertices.tolist()
                hull.append(hull[0])
                plt.plot(cluster[hull, 0], cluster[hull, 1], 'c--')
            except:
                pass
        
        # 初始的cluster_center特殊标注一下
        plt.scatter(self.initial_copied[:, 0], self.initial_copied[:, 1], c='#00FF00',edgecolors='#FF0000',linewidths=2)
        plt.pause(0.5)

    def __generate_Center(self, cluster: np.ndarray):
        center=[]
        for property_index in range(cluster.shape[1]):
            column=cluster[:,property_index]
                        
            # 连续值
            if np.issubdtype(type(column[0]),np.number):
                # 均值即是cluster_center的property_value
                center.append( np.mean(column, axis=0) )
            # 离散值
            else:
                # 这里统计cluster内的同种property下的所有property_value，为每一种property_value计算权重
                # 将[(property_value1,0.2),(property_value2,0.3),(property_value3,0.5)]作为cluster_center的property_value
                temp=[]
                for property_value in np.unique(column):
                    temp.append( (property_value, len(column[column==property_value]) / len(column)) )
                center.append(temp)
        return center

    def __distance_Value_Value(self, property_index: int, Va, Vb) -> float:
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
                distance+=(muai/mua+mubi/mub)**2
        
        return distance**0.5

    def __distance_Sample_Sample(self, Sa:np.ndarray, Sb: np.ndarray) -> float:
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
            distance+=self.__distance_Value_Value(i,Sa[i],Sb[i])
        
        return distance

    def __distance_Sample_Center(self, x:np.ndarray, cluster_center) -> float:
        """样本x与簇中心的距离

        Args:
            x (np.ndarray): 样本x
            cluster_center ([type]): self.cluster_center_list中的元素

        Returns:
            float: 样本x与簇中心的距离
        """
        
        distance=0
        
        # 遍历每一种property，计算x与cluster_center的distance
        for property_index in range(len(x)):
            
            property_value=x[property_index]
            cluster_center_property_value=cluster_center[property_index]
            if np.issubdtype(type(property_value),np.number):
                distance+=self.__distance_Value_Value(property_index,property_value,cluster_center_property_value)                
            else:
                for thing in cluster_center_property_value:
                    value=thing[0]
                    weight=thing[1]
                    distance+=self.__distance_Value_Value(property_index,property_value,value)*weight
        
        return distance
    
    def __distance_Center_Center(self, cluster_index1: int, cluster_index2: int) -> float:
        """簇与簇中心的距离

        Args:
            cluster_index1 (int): self.cluster_center_list中的下标
            cluster_index2 (int): self.cluster_center_list中的下标

        Returns:
            float: 簇与簇中心的距离
        """

        center1=self.cluster_center_list[cluster_index1]
        center2=self.cluster_center_list[cluster_index2]
        if len(center1)!=len(center2):
            raise ValueError("len(center1)!=len(center2)!")
        
        dist=0
        for property_index in range(len(center1)):
            cluster_center_property_value1=center1[property_index]
            cluster_center_property_value2=center2[property_index]
            if np.issubdtype(type(cluster_center_property_value1),np.number):
                dist+=self.__distance_Value_Value(property_index,cluster_center_property_value1,cluster_center_property_value2)                
            else:
                for thing1 in cluster_center_property_value1:
                    value1=thing1[0]
                    weight1=thing1[1]
                    for thing2 in cluster_center_property_value2:
                        value2=thing2[0]
                        weight2=thing2[1]
                        dist+=self.__distance_Value_Value(property_index,value1,value2)*weight2*weight1
        return dist

    def __dist_max_neighbor(self, cluster: np.ndarray) -> float:
        """簇内最大的邻居距离

        Args:
            cluster (np.ndarray): cluster

        Returns:
            float: 簇内最大的邻居距离
        """
        if len(cluster)==1:
            return float('inf')

        l=[]
        for i in range(len(cluster)):
            sa=cluster[i]
            min_dist=float('inf')
            for j in range(len(cluster)):
                if j!=i:
                    sb = cluster[j]
                    dist = self.__distance_Sample_Sample(sa,sb)
                    if dist<min_dist:
                        min_dist=dist
            l.append(min_dist)
        return max(l)

    def __dist_neighbor(self, cluster: np.ndarray) -> float:
        """簇内平均的邻居距离

        Args:
            cluster (np.ndarray): cluster

        Returns:
            float: 簇内平均的邻居距离
        """
        if len(cluster)==1:
            return float('inf')

        l=[]
        for i in range(len(cluster)):
            sa=cluster[i]
            min_dist=float('inf')
            for j in range(len(cluster)):
                if j!=i:
                    sb = cluster[j]
                    dist = self.__distance_Sample_Sample(sa,sb)
                    if dist<min_dist:
                        min_dist=dist
            l.append(min_dist)
        l=np.array(l)
        return l.mean()

    def __dist_sum(self, cluster: np.ndarray) -> float:
        """簇内的距离和

        Args:
            cluster (np.ndarray): cluster

        Returns:
            float: 簇内的距离和
        """        

        dist=0
        for i in range(len(cluster)-1):
            sa=cluster[i]
            for j in range(i+1,len(cluster)):
                sb = cluster[j]
                dist += self.__distance_Sample_Sample(sa,sb)
        return dist
    
    def __dist_avg(self, cluster: np.ndarray) -> float:
        """簇内的平均距离

        Args:
            cluster (np.ndarray): cluster

        Returns:
            float: 簇内的平均距离
        """        

        return 2/(len(cluster)+1e-16)/(len(cluster)-1+1e-16)*self.__dist_sum(cluster)
    
    def __dist_var(self, cluster: np.ndarray) -> float:
        """簇内的方差

        Args:
            cluster (np.ndarray): cluster

        Returns:
            float: 簇内的方差
        """        
        center=self.__generate_Center(cluster)
        dist=0
        for i in cluster:
            dist+=self.__distance_Sample_Center(i,center)
        
        return dist/len(cluster)

    def __dist_diam(self, cluster: np.ndarray) -> float:
        """簇的直径（簇内的最大距离）

        Args:
            cluster (np.ndarray): cluster

        Returns:
            float: 簇的直径（簇内的最大距离）
        """        
        
        max_dist=0
        for i in range(len(cluster)-1):
            sa=cluster[i]
            for j in range(i+1,len(cluster)):
                sb=cluster[j]
                dist=self.__distance_Sample_Sample(sa,sb)
                if dist>max_dist:
                    max_dist=dist
        return max_dist
    
    def __dist_edge(self, cluster1: np.ndarray, cluster2: np.ndarray) -> float:
        """簇间最短的边缘距离

        Args:
            cluster1 (np.ndarray): cluster1
            cluster2 (np.ndarray): cluster2

        Returns:
            float: 簇间最短的边缘距离
        """        
        
        min_dist=float('inf')
        for sa in cluster1:
            for sb in cluster2:
                dist=self.__distance_Sample_Sample(sa,sb)
                if dist<min_dist:
                    min_dist=dist
        return min_dist
    
    def score(self, method: Literal['DBI', 'DI', 'WSS']):
        
        if method=='DBI':
            # 越小越好
            sum=0
            for i in range(self.k):
                max=0
                for j in range(self.k):
                    if i!=j:
                        cluster1=self.sample_list[self.sample_cluster_index_list==i]
                        cluster2=self.sample_list[self.sample_cluster_index_list==j]
                        res=(self.__dist_avg(cluster1)+self.__dist_avg(cluster2)) / (self.__distance_Center_Center(i,j)+1e-16)
                        if res>max:
                            max=res
                sum+=max
            
            sum=sum/self.k

            return sum

        elif method=='DI':
            # 越大越好
            max_diam=0
            for i in range(self.k):
                cluster=self.sample_list[self.sample_cluster_index_list==i]
                res=self.__dist_diam(cluster)
                if res>max_diam:
                    max_diam=res
            
            min_d_min=float('inf')
            for i in range(self.k):
                for j in range(i+1,self.k):
                    cluster1=self.sample_list[self.sample_cluster_index_list==i]
                    cluster2=self.sample_list[self.sample_cluster_index_list==j]
                    res=self.__dist_edge(cluster1,cluster2)
                    if res<min_d_min:
                        min_d_min=res
            
            return min_d_min/(max_diam+1e-16)
        
        elif method=='WSS':
            # 越小越好
            sum=0
            for i in range(self.k):
                cluster=self.sample_list[self.sample_cluster_index_list==i]
                sum+=self.__dist_avg(cluster)
            return sum
        else:
            raise ValueError("method should be in ['DBI', 'DI', 'WSS]")