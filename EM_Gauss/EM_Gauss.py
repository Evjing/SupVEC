# -*- coding:utf-8 -*-
#############################################
#
#      EM算法高斯混合聚类模型
#      利用sklearn的make_blobs方法生成高斯混合数据集
#      使用sklearn的k均值聚类方法得到聚类模型的初始参数
#      采用EM算法进行迭代，并绘出迭代过程中分类的结果以及聚类中心
#      按键'q'可强制中断迭代过程
#      最后输出最终的分类结果以及聚类中心，并和实际的类别标签进行比对
#
#############################################
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from sklearn.cluster import KMeans
import numpy as np
import pickle
import random
# 导入数据
data_num = 300
features_num = 2
target_num  = 7

Data,Target=make_blobs(n_samples=data_num,n_features=features_num,centers=target_num)

# #存储和读取随机参数
# pickle.dump(Data,open('.\\Data.txt', 'wb') ) 
# pickle.dump(Target,open('.\\Target.txt', 'wb') ) 
# Data = pickle.load(open('.\\Data.txt', 'rb') ) 
# Target = pickle.load(open('.\\Target.txt', 'rb') ) 

class Em_Gauss_Model:
    def Em_Gauss_Model(self):
        pass
    def Em_Gauss_Model(self,Data,target_num):
        self.init(Data,target_num)

    def init(self,Data,target_num):
        ### EM算法 高斯混合概形
        # 初始化模型参数
        self.Y = np.array(Data)
        self.target_num = target_num
        self.data_num = self.Y.shape[0]
        self.features_num = self.Y.shape[1]
        self.P = np.ones(target_num) /target_num # 类别概率
        self.U = np.zeros([target_num,self.features_num])   # 高斯分布均值参数
        self.Sigma = np.zeros([target_num,self.features_num,self.features_num])   # 高斯分布协方差参数
        self.Tji = np.zeros([target_num,self.data_num])
        self.fji =  np.zeros([target_num,self.data_num])
        self.Target_predict = np.zeros(self.data_num) 
        self.init_theta('sklearn-Kmeans')


    # 参数初始化,供外部算法提供初始参数
    def init_theta(self,mode = 'sklearn-Kmeans'):
        # 初始化均值
        if mode == 'sklearn-Kmeans':
            # 使用sklean库的K均值进行初始化
            kmeans = KMeans(n_clusters=self.target_num).fit(self.Y)
            y_pred =kmeans.fit_predict(self.Y)
            self.U_init =kmeans.cluster_centers_
            pass
        elif mode == 'random':
            # 随机初始化
            self.U_init = self.Y[np.random.randint(0,self.data_num,self.target_num)]
            pass
        self.U  = self.U_init
        # 初始化协方差
        Sigma_init = self.Sigma.copy()
        for j in range(target_num):
            Sigma_init[j] = 2*features_num*np.eye(self.features_num)
        pass
        self.Sigma = Sigma_init


    def cal_gauss_f(self,U_j,Sigma_j,Y):
        # 计算高斯密度
        Y_U_j = (Y-U_j).reshape([features_num,1])  # 维数 1xfeatures_num
        f_Y_j = np.dot(Y_U_j.T,np.linalg.inv(Sigma_j)) # 维数 1xfeatures_num
        f_Y_j_up = -np.dot(f_Y_j,Y_U_j)/2 # 维数 1x1
        f_Y_j_full  = np.exp(f_Y_j_up) / ( np.power(2*np.pi,features_num/2)  * np.sqrt(np.abs(np.linalg.det(Sigma_j)))) # 求得高斯密度
        ### 防止密度溢出或者为0
        if np.isnan(f_Y_j_full):
            f_Y_j_full = 100000000
        if f_Y_j_full == 0:
            f_Y_j_full = 0.00000000001
        return f_Y_j_full

    # 计算Tji
    def cal_T_i_j(self):
        for j in range(self.target_num):
            for i in range(self.data_num):
                self.fji[j][i] = self.cal_gauss_f(self.U[j],self.Sigma[j],self.Y[i])
        P_f = np.dot(self.P.reshape([1,self.target_num]),self.fji)
        P_f = P_f.flatten()
        for j in range(self.target_num):
            for i in range(self.data_num):
                # 防止微小量的影响
                self.Tji[j][i] = self.P[j] * self.fji[j][i]/P_f[i]

    def cal_P_i_j(self):
        Tj_sum = np.sum(self.Tji,axis=1)
        self.P = Tj_sum/self.data_num
        # print(P)

    def cal_theta_i_j(self):
        U_new = np.zeros([self.target_num,self.features_num]) 
        Sigma_new =np.zeros([self.target_num,self.features_num,self.features_num]) ;
        Tj_sum = np.sum(self.Tji,axis=1)
        T_Q = np.array(self.Tji);
        # 更新均值
        for j in range(self.target_num):
            U_j = np.zeros([1,features_num])
            for i in range(self.data_num):
                U_j = U_j + self.Tji[j][i]*self.Y[i]
            U_j = U_j/Tj_sum[j]
            U_new[j] = U_j
        # 更新协方差
        for j in range(self.target_num):
            Sigma_j = np.zeros([self.features_num,self.features_num])
            for i in range(self.data_num):
                Y_U_j = self.Y[i] -U_new[j]
                Y_U_j = Y_U_j.reshape([self.features_num,1])
                Sigma_j = Sigma_j + self.Tji[j][i]*np.dot(Y_U_j,Y_U_j.T)
            Sigma_j = Sigma_j/Tj_sum[j]
            Sigma_new[j] = Sigma_j
        #print(U_new-U)
        return U_new,Sigma_new
    def on_key_press(self,event):
        if(event.key == 'q'):
            self.key_break = 1
    def train(self):
        stop_pre = 0
        stop_count = 0
        max_count = 30
        stop_deta = 0.001
        fig, ax = pyplot.subplots()
        self.key_break = 0
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        L_theta_pre = -np.inf;
        iter_count = 0

        while(True):
            self.cal_T_i_j()
            self.cal_P_i_j()
            U_new,Sigma_new = self.cal_theta_i_j()
            stop_q = np.sqrt(np.sum(np.square(self.U-U_new)) + np.sum(np.square(self.Sigma - Sigma_new)))

            # 计算似然函数.
            L_theta = 0
            for i in range(self.data_num):
                L_theta_i = 0;
                for j in range(self.target_num):
                    L_theta_i = L_theta_i + self.Tji[j][i] * self.P[j]
                L_theta_i = np.log(L_theta_i)
                L_theta = L_theta + L_theta_i
            print("iter ",end="")
            print(iter_count ,end=" :")
            print("    参数变化：  ",end="")
            print(stop_q ,end="")
            print("    似然函数：  ",end="")
            print(L_theta)

            # 参数不变化退出
            if(stop_q <stop_deta):
                print("Normal Break")
                break
            # 似然函数增大一定次数后退出
            if(L_theta_pre >= L_theta):
                stop_count = stop_count + 1
                if stop_count > max_count:
                    print("Un Normal Break")
                    break 



            # 按键退出
            if(self.key_break == 1):
                print("Key Break")
                break

            # 更新
            self.U = U_new.copy()
            self.Sigma = Sigma_new.copy()
            self.Target_predict  = np.argmax(self.Tji,0)
            L_theta_pre = L_theta
            stop_pre = stop_q
            iter_count = iter_count + 1

            # 显示聚类迭代过程
            pyplot.clf()
            pyplot.title('EM算法迭代过程',fontproperties='SimHei')
            pyplot.scatter(self.Y[:,0],self.Y[:,1],c= self.Target_predict,marker='.')
            pyplot.scatter(self.U[:,0],self.U[:,1],c='r',marker='o');
            pyplot.pause(0.001)
           


# 绘制效果
def my_plot(Data,Target,U_init,Y,Target_predict,U):
    pyplot.clf()
    # 在2D图中绘制样本，每个样本颜色不同
    pyplot.subplot(121)
    pyplot.title('skLearn生成高斯混合样本',fontproperties='SimHei')
    pyplot.scatter(Data[:,0],Data[:,1],c=Target,marker='.')
    pyplot.scatter(U_init[:,0],U_init[:,1],c='r',marker='o');

    # 在2D图中绘制样本，每个样本颜色不同
    pyplot.subplot(122)
    pyplot.title('EM算法估计GMM分布结果',fontproperties='SimHei')
    pyplot.scatter(Y[:,0],Y[:,1],c= Target_predict,marker='.')
    pyplot.scatter(U[:,0],U[:,1],c='r',marker='o');
    pyplot.pause(0.001)
    pyplot.show()



def main():
    model = Em_Gauss_Model()
    model.init(Data,target_num)
    model.train()
    my_plot(Data,Target, model.U_init,model.Y,model.Target_predict,model.U)

if __name__ == '__main__':
    main()
    # print(__name__)
