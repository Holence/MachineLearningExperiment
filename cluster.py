import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import random
from scipy.spatial import ConvexHull
from utils import logistic

class Cluster():
    def __init__(self, k: int = 2) -> None:
        self.k = k
        self.cluster_center_list = [] # 簇的中心点
        self.sample_list = np.array([]) # 样本
        self.sample_cluster_index_list = np.array([]) # 样本对应的簇的下标
        
        self.initial_centers = None
        self.colors = []

    def fit(self, X: np.ndarray, max_iters: int = None, plt_process: bool = False):
        """聚类

        Args:
            X (np.ndarray): X
            max_iters (int, optional): 最大迭代次数。如果为None则一直迭代到中心点不再变化才停止. Defaults to None.
            plt_process (bool, optional): 是否画图，前提是输入的样本特征维度为2. Defaults to False.
        """

        if max_iters!=None and max_iters<=0:
            raise ValueError("max_iters should larger than zero!")
        
        if plt_process:
            # 一次fit一种配色
            self.colors=[]
            for i in range(self.k):
                self.colors.append(np.random.random((1,1,3)))
        
        # 每次fit都会将X保存至sample_list
        self.sample_list=X

    def predict(self, x: np.ndarray):
        """预测单条数据

        Args:
            x (np.ndarray): 单条x数据
        """
        
        pass
    
    def plot2D_Clusters(self, cluster_list: list, pause: float = 0.5):
        """绘制特定维度的2维散点图

        Args:
            cluster_list ([type]): 多簇二维的数据，比如np.array([ [ [1.1,1.2], [1.3,1.1] ] , [ [2.8,2.9], [2.5,2.6] ] ])
            pause (float, optional): 绘图暂停的时间. Defaults to 0.5.
        """

        if self.colors==[]:
            for i in range(self.k):
                self.colors.append(np.random.random((1,1,3)))
        
        plt.cla() # 清除原有图像
        plt.title("cluster nums: %d"%len(cluster_list))

        i=0
        for cluster in cluster_list:
            plt.scatter(cluster[:, 0], cluster[:, 1], c=self.colors[i])
            
            # 画出每个簇的凸包
            try:
                hull = ConvexHull(cluster).vertices.tolist()
                hull.append(hull[0])
                plt.plot(cluster[hull, 0], cluster[hull, 1], 'c--')
            except:
                pass
            i+=1

        if pause!=0:
            plt.pause(pause)
    
    def plot2D(self, plt_initial_center: bool = True, pause: float = 0.5):
        """如果样本的特征维度就是二维，则直接调用这个函数绘制2维散点图

        Args:
            plt_initial_center (bool, optional): 是否标注初始中心点. Defaults to True.
            pause (float, optional): 绘图暂停的时间. Defaults to 0.5.
        """

        cluster_list=[]
        for i in range(self.k):
            cluster = self.sample_list[self.sample_cluster_index_list==i]
            cluster_list.append(cluster)
        cluster_list=np.array(cluster_list,dtype=object)

        self.plot2D_Clusters(cluster_list, 0)
        if plt_initial_center and type(self.initial_centers)!=type(None):
            plt.scatter(self.initial_centers[:, 0], self.initial_centers[:, 1], c='#00FF00',edgecolors='#FF0000',linewidths=2)
        
        if pause!=0:
            plt.pause(pause)
        
    def _generate_Center(self, cluster: np.ndarray):
        """计算cluster的中心点，连续值属性的中心为平均值，离散值属性的中心为附带权重的列表

        Args:
            cluster (np.ndarray): 多条含有 多维特征 的样本

        Returns:
            [type]: cluster的中心点，比如[0.1, 0.2, [('white', 0.8), ('black', 0.2)] ]
        """        
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

    def _distance_Sample_Center(self, x:np.ndarray, cluster_center) -> float:
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
                distance+=self._distance_Value_Value(property_index,property_value,cluster_center_property_value)                
            else:
                for thing in cluster_center_property_value:
                    value=thing[0]
                    weight=thing[1]
                    distance+=self._distance_Value_Value(property_index,property_value,value)*weight
        
        return distance
    
    def _distance_Center_Center(self, cluster_index1: int, cluster_index2: int) -> float:
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
                dist+=self._distance_Value_Value(property_index,cluster_center_property_value1,cluster_center_property_value2)                
            else:
                for thing1 in cluster_center_property_value1:
                    value1=thing1[0]
                    weight1=thing1[1]
                    for thing2 in cluster_center_property_value2:
                        value2=thing2[0]
                        weight2=thing2[1]
                        dist+=self._distance_Value_Value(property_index,value1,value2)*weight2*weight1
        return dist

    def _dist_max_neighbor(self, cluster: np.ndarray) -> float:
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
                    dist = self._distance_Sample_Sample(sa,sb)
                    if dist<min_dist:
                        min_dist=dist
            l.append(min_dist)
        return max(l)

    def _dist_neighbor(self, cluster: np.ndarray) -> float:
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
                    dist = self._distance_Sample_Sample(sa,sb)
                    if dist<min_dist:
                        min_dist=dist
            l.append(min_dist)
        l=np.array(l)
        return l.mean()

    def _dist_sum(self, cluster: np.ndarray) -> float:
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
                dist += self._distance_Sample_Sample(sa,sb)
        return dist
    
    def _dist_avg(self, cluster: np.ndarray) -> float:
        """簇内的平均距离

        Args:
            cluster (np.ndarray): cluster

        Returns:
            float: 簇内的平均距离
        """

        return 2/(len(cluster)+1e-16)/(len(cluster)-1+1e-16)*self._dist_sum(cluster)
    
    def _dist_var(self, cluster: np.ndarray) -> float:
        """簇内的方差

        Args:
            cluster (np.ndarray): cluster

        Returns:
            float: 簇内的方差
        """

        center=self._generate_Center(cluster)
        dist=0
        for i in cluster:
            dist+=self._distance_Sample_Center(i,center)
        
        return dist/len(cluster)

    def _dist_diam(self, cluster: np.ndarray) -> float:
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
                dist=self._distance_Sample_Sample(sa,sb)
                if dist>max_dist:
                    max_dist=dist
        return max_dist
    
    def _dist_edge(self, cluster1: np.ndarray, cluster2: np.ndarray) -> float:
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
                dist=self._distance_Sample_Sample(sa,sb)
                if dist<min_dist:
                    min_dist=dist
        return min_dist
    
    def score(self, method: Literal['DBI', 'DI', 'WSS']):
        """性能度量的指标

        Args:
            method (Literal['DBI', 'DI', 'WSS']): 可选的性能度量指标

        Returns:
            [type]: 性能度量的指标
        """        
        if method=='DBI':
            # 越小越好
            sum=0
            for i in range(self.k):
                max=0
                for j in range(self.k):
                    if i!=j:
                        cluster1=self.sample_list[self.sample_cluster_index_list==i]
                        cluster2=self.sample_list[self.sample_cluster_index_list==j]
                        res=(self._dist_avg(cluster1)+self._dist_avg(cluster2)) / (self._distance_Center_Center(i,j)+1e-16)
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
                res=self._dist_diam(cluster)
                if res>max_diam:
                    max_diam=res
            
            min_d_min=float('inf')
            for i in range(self.k):
                for j in range(i+1,self.k):
                    cluster1=self.sample_list[self.sample_cluster_index_list==i]
                    cluster2=self.sample_list[self.sample_cluster_index_list==j]
                    res=self._dist_edge(cluster1,cluster2)
                    if res<min_d_min:
                        min_d_min=res
            
            return min_d_min/(max_diam+1e-16)
        
        elif method=='WSS':
            # 越小越好
            sum=0
            for i in range(self.k):
                cluster=self.sample_list[self.sample_cluster_index_list==i]
                sum+=self._dist_avg(cluster)
            return sum
        else:
            raise ValueError("method should be in ['DBI', 'DI', 'WSS]")

class KMeans(Cluster):
    def __init__(self, k: int = 2) -> None:
        super().__init__(k=k)
    
    def fit(self, X: np.ndarray, initial_center_indexs: list = None, max_iters: int = None, plt_process: bool = False):
        super().fit(X, max_iters=max_iters, plt_process=plt_process)
        
        # 选定初始点，先随机选一个样本，再生成k-1个
        # 选择的条件是与initial_center_indexs中已选点的距离和最大 且 与每个已选点的距离不小于0.9*avg_dist
        
        # 如果含有离散值，则由于没有分类，所以一开始无法计算样本间的距离，这里直接把sample_cluster_index_list全部设为一类，离散属性值间的距离就全部为0，对计算距离起作用的只有连续值了
        self.sample_cluster_index_list=np.zeros(len(self.sample_list))
        
        if initial_center_indexs==None:
            initial_center_indexs=[random.randint(0,self.sample_list.shape[0]-1)]
            avg_dist=self._dist_avg(self.sample_list)
            for i in range(self.k-1):
                max_dist=0
                max_dist_index=None
                for Sb_index in range(self.sample_list.shape[0]):
                    if Sb_index not in initial_center_indexs:
                        Sb=self.sample_list[Sb_index]
                        dist=0
                        flag=True
                        for Sa_index in initial_center_indexs:
                            Sa=self.sample_list[Sa_index]
                            res=self._distance_Sample_Sample(Sa,Sb)
                            if res<avg_dist*0.9:
                                flag=False
                                break
                            else:
                                dist+=res
                        
                        if flag==True and dist>max_dist:
                            max_dist=dist
                            max_dist_index=Sb_index
                if max_dist_index==None:
                    max_dist_index=initial_center_indexs[0]
                    while max_dist_index in initial_center_indexs:
                        max_dist_index=random.randint(0,self.sample_list.shape[0]-1)
                initial_center_indexs.append(max_dist_index)
        else:
            if len(initial_center_indexs)!=self.k:
                raise ValueError("len(initial_center_indexs)!=self.k")
        
        # 用initial_center_indexs初始化self.cluster_center_list
        initial_center_indexs=self.sample_list[initial_center_indexs]
        self.cluster_center_list=[]
        for i in initial_center_indexs:
            cluster=np.array([i])
            self.cluster_center_list.append(self._generate_Center(cluster))

        self.initial_centers=np.array(self.cluster_center_list.copy(),dtype='object')

        while True:
            
            # 计算每个sample所属的cluster，并记录下对应的下标，存储到sample_cluster_index_list中
            self.sample_cluster_index_list = []
            for sample in self.sample_list:
                # 按照中心点给样本分类
                self.sample_cluster_index_list.append(self.predict(sample))
            self.sample_cluster_index_list=np.array(self.sample_cluster_index_list)
            
            # 如果是plt_process，则画图
            if plt_process and self.sample_list.shape[1]==2:
                self.plot2D()                
            
            # 更新cluster_center
            old_center_list=self.cluster_center_list.copy()
            self.cluster_center_list=[]
            for i in range(self.k):
                cluster=self.sample_list[self.sample_cluster_index_list==i]
                self.cluster_center_list.append(self._generate_Center(cluster))
            
            # 如果没有更新了，退出
            if self.cluster_center_list==old_center_list:
                break
            
            # 如果设定了迭代次数
            if max_iters!=None:
                max_iters-=1
                # 迭代次数到了，退出
                if max_iters==0:
                    break
    
    def predict(self, x: np.ndarray, retrun_index: bool = True):
        """距离哪个中心点最近就归属于哪一簇

        Args:
            x (np.ndarray): x
            retrun_index (bool, optional): 返回簇的下标，还是返回簇中心点. Defaults to True.

        Returns:
            [type]: 簇的下标或者簇中心点
        """

        min_distance=float('inf')
        min_cluster_index=None
        # 计算样本x与每一个cluster_center的distance，选出最小的一个作为归属
        for cluster_index in range(self.k):
            
            distance=self._distance_Sample_Center(x,self.cluster_center_list[cluster_index])

            if distance<min_distance:
                min_distance=distance
                min_cluster_index=cluster_index
        
        if retrun_index:
            return min_cluster_index
        else:
            return self.cluster_center_list[min_cluster_index]

class AutoKMeans(KMeans):
    def __init__(self) -> None:
        super().__init__(k=2)
    
    def fit(self, X: np.ndarray):
        # elbow method
        while True:
            l=[]
            for i in range(10):
                super().fit(X, initial_center_indexs=None, max_iters=None, plt_process=False)
                s=self.score('DBI')
                l.append(s)
            s=np.mean(l)
            
            if self.k>3:
                if old_s-s<delta or old_s-s<0:
                    self.k-=1
                    break
                else:
                    delta=s-old_s
                    old_s=s
            if self.k==3:
                if old_s-s<0:
                    self.k-=1
                    break
                else:
                    delta=old_s-s
                    old_s=s
            if self.k==2:
                old_s=s

            self.k+=1
        
        # 确定k后，再跑个几次，选聚类效果最好的出来
        min=float('inf')
        for i in range(10):
            super().fit(X, initial_center_indexs=None, max_iters=None, plt_process=False)
            s=self.score('DBI')
            
            if s<min:
                min=s
                best_cluster_center_list=self.cluster_center_list
                best_sample_cluster_index_list=self.sample_cluster_index_list
                best_initial_centers=self.initial_centers
        
        self.cluster_center_list=best_cluster_center_list
        self.sample_cluster_index_list=best_sample_cluster_index_list
        self.initial_centers=best_initial_centers


class AutoClustering(Cluster):
    def __init__(self) -> None:
        super().__init__(k=1)
    
    def fit(self, X: np.ndarray, max_iters: int = None, plt_process: bool = False):
        super().fit(X, max_iters=max_iters, plt_process=plt_process)
        
        self.k=1
        
        # 这里初始化每个样本的归属，全部设为一簇
        self.sample_cluster_index_list = np.zeros(self.sample_list.shape[0])
        
        # 画一下图
        if plt_process and self.sample_list.shape[1]==2:
            self.plot2D()

        self.splitted_cluster=[]
        # 根据样本分布，自动增减k值
        while True:
            
            flag=False

            # 簇太多了吗？应该合并
            i=0
            while i<self.k and self.k>2:
                # 以i簇为起点，向其他簇试探
                cluster1 = self.sample_list[self.sample_cluster_index_list==i]
                neighbor1=self._dist_max_neighbor(cluster1)
                
                j=i+1
                while j<self.k and self.k>2:
                    cluster2 = self.sample_list[self.sample_cluster_index_list==j]
                    neighbor2=self._dist_max_neighbor(cluster2)
                    edge=self._dist_edge(cluster1,cluster2)

                    # 如果 i簇与j簇的 边缘距离 小于 各自簇内的邻居距离，则合并
                    if edge<=neighbor1*1.2 or edge<=neighbor2*1.2:

                        flag=True

                        # cluster2并入cluster1
                        self.sample_cluster_index_list[self.sample_cluster_index_list==j] = i
                        
                        if i in self.splitted_cluster:
                            self.splitted_cluster.append(i)
                        
                        if j in self.splitted_cluster:
                            self.splitted_cluster=[t for t in self.splitted_cluster if t!=j]

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
                            self.plot2D()
                    
                    j+=1
            
                i+=1
            
            # 簇太少了吗？应该分裂
            i=0
            one_more_try=False
            while i<self.k:
                if plt_process and self.sample_list.shape[1]==2:
                    self.plot2D()
                
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
                
                
                full_neighbor=self._dist_max_neighbor(cluster)
                
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
                        # 等取到一定样本个数后，threshold变成_dist_max_neighbor
                        else:
                            cc=self.sample_list[[origin_index_list[done] for done in done_list]]
                            neighbor=self._dist_max_neighbor(cc)
                            threshold = neighbor
                        
                        adding=[ left for left in left_list if self._distance_Sample_Sample(cluster[checking],cluster[left]) < threshold]
                        
                        # 如果步幅仍太小，有一次提升步幅的机会
                        if adding==[]:
                            cc=self.sample_list[[origin_index_list[done] for done in done_list]]
                            var = 1 + logistic(self._dist_var(cc)**2)
                            # avg=1+logistic(self._dist_avg(cc))
                            # threshold = neighbor + avg*var
                            threshold *= var
                            adding=[ left for left in left_list if self._distance_Sample_Sample(cluster[checking],cluster[left]) < threshold]
                        
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
                    
                    edge=self._dist_edge(dd,ll)
                    neighbor_ll=self._dist_max_neighbor(ll)
                    neighbor_dd=self._dist_max_neighbor(dd)
                    if edge>neighbor_ll and edge>neighbor_dd:
                        
                        flag=True

                        # 给left_list中分配新的cluster_index
                        self.sample_cluster_index_list[left_list]=self.k

                        self.splitted_cluster.append(i)

                        self.k+=1
                        if plt_process:
                            # 颜色的个数也要重新调整
                            self.colors.append(np.random.random((1,1,3)))
                        
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
            if flag==False:
                break

            # 如果设定了迭代次数
            if max_iters!=None:
                max_iters-=1
                # 迭代次数到了，退出
                if max_iters==0:
                    break
        
        # 最后计算一下重心，虽然predict中不会用到，但在计算score中会用到
        self.cluster_center_list=[]
        for i in range(self.k):
            cluster=self.sample_list[self.sample_cluster_index_list==i]
            self.cluster_center_list.append(self._generate_Center(cluster))
    
    def predict(self, x: np.ndarray, retrun_index: bool = True):
        """距离哪个簇最近就归属于哪一簇

        Args:
            x (np.ndarray): x
            retrun_index (bool, optional): 返回簇的下标，还是返回簇的所有样本. Defaults to True.

        Returns:
            [type]: 簇的下标或者簇的所有样本
        """
        min=float('inf')
        closest_cluster_index=None
        for i in range(self.k):
            cluster = self.sample_list[self.sample_cluster_index_list==i]
            for p in cluster:
                d=self._distance_Sample_Sample(p,x)
                if d<min:
                    min=d
                    closest_cluster_index=i
        
        if retrun_index:
            return closest_cluster_index
        else:
            return self.sample_list[self.sample_cluster_index_list==closest_cluster_index]