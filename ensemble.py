import numpy as np
import matplotlib.pyplot as plt
from typing import Type
from collections import namedtuple
from learner import Learner
from metrics import accuracy_score

Learner_and_Weight_Tuple=namedtuple("Learner_and_Weight",["learner","weight"])

class AdaBoost():
    def __init__(self, times: int, learner: Type[Learner], *learner_args, stop_in_advance = False) -> None:
        """AdaBoost构造函数

        Args:
            times (int): 集成学习的次数（集成学习器的个数）
            learner (Type[Learner]): 学习器（是class的地址，不是instance的地址）
            stop_in_advance (bool, optional): 判断综合预测能力是否到达100%，若到达则提前结束，可以预防过拟合. Defaults to False.
        """        
        self.times=times
        self.learner=learner
        self.learner_args=learner_args
        self.stop_in_advance=stop_in_advance
        self.boosted_learner_list=[] # list[Learner_and_Weight_Tuple] 学习器列表，存放学习器learner和权重a
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """AdaBoost训练，输入X、Y、集成学习器的个数、要使用的学习器、学习器附加的参数，输出学习器列表

        Args:
            X (np.ndarray): X
            Y (np.ndarray): Y
        """
        
        self.boosted_learner_list=[]

        n=Y.shape[0]
    
        sample_weight = np.ones(n) / n
        err = 1
        
        print("------AdaBoost Train Started------")
        
        time=1
        err_list=[]
        error_rate_list=[]
        full_error_rate_list=[]
        
        while time<=self.times:
            print("Boosting Train Round %s"%time)
            
            temp_learner=self.learner(*self.learner_args)
            temp_learner.fit(X,Y,sample_weight)

            predict_y=temp_learner.predictBatch(X)
            # err权重和
            err=np.sum(sample_weight[predict_y!=Y])
            print("\t\tCurrent_Err:",err)
            err_list.append(err)
            
            # 单个学习器的错误率
            error_rate=1-accuracy_score(Y,predict_y)
            print("\t\tCurrent_Error_rate:",error_rate)
            error_rate_list.append(error_rate)

            a = np.log((1-err)/max(err,1e-16))/2
            
            # 万一y不只是二分类（即不只是1和-1），就需要手动操作一下
            same_or_not = np.ones(n)
            same_or_not[predict_y==Y]=-1

            sample_weight = sample_weight*np.exp(a*same_or_not)
            sample_weight = sample_weight/np.sum(sample_weight)

            self.boosted_learner_list.append(Learner_and_Weight_Tuple(temp_learner,a))
            
            
            full_predict_y=self.predictBatch(X)
            # 综合学习器的错误率
            full_error_rate=1-accuracy_score(Y,full_predict_y)
            print("\t\tFull_Current_Error_rate:", full_error_rate)
            full_error_rate_list.append(full_error_rate)
            
            if err>=0.5:
                print("\n\n哦我的上帝，err>=0.5，学习到此为止\n\n你可以检查一下数据集了\n或者提升一下学习器的表征能力\n或者换一种学习器，该学习器的学习能力已经达到极限了\n\n")
                break
            
            if self.stop_in_advance:
                if full_error_rate==0:
                    print("\n\n哦我的上帝，仅仅只用了%s次，综合预测能力就已经100%%正确了，学习到此为止\n\n"%time)
                    break
            
            if err<=1e-4:
                print("\n\n哦我的上帝，仅仅只用了%s次，err<=1e-4，学习到此为止\n\n"%time)
                break
            
            time+=1
        
        if time>self.times:
            time=self.times
        
        print("------AdaBoost Train Finished------\n")

        plt.figure(figsize=(10,8))
        plt.xticks(np.arange(0,time,1))
        plt.title(f"Adaboost\n{self.learner.__name__} {self.learner_args}\nRun stops at {time}/{self.times}")
        plt.plot(err_list)
        plt.plot(error_rate_list)
        plt.plot(full_error_rate_list)
        plt.legend(["Err","Error Rate","Full Error Rate"])

        plt.show()

    def predict(self, x: np.ndarray):
        """AdaBoost单条预测，传入x

        Args:
            x (np.ndarray): x

        Returns:
            [type]: 预测的y值
        """

        votes={}
        for learner_and_weight_Tuple in self.boosted_learner_list:
            a=learner_and_weight_Tuple.weight
            learner=learner_and_weight_Tuple.learner
            res=learner.predict(x)
            if votes.get(res)==None:
                votes[res]=a
            else:
                votes[res]+=a
        
        votes=sorted(votes.items(),key=lambda x:x[1],reverse=True)
        return votes[0][0]
    
    def predictBatch(self, X: np.ndarray) -> np.ndarray:
        """AdaBoostPredictSingle的批量版，传入X

        Args:
            X (np.ndarray): X

        Returns:
            np.ndarray: 预测的y值序列
        """

        res=[]
        for i in range(X.shape[0]):
            res.append(self.predict(X[i,:]))
        return np.array(res)
