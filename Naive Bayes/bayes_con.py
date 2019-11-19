# -*-coding:utf-8 -*-

"""

本程序构造实现了连续型的朴素贝叶斯算法

使用Iris数据集作为测试用以验证算法的有效性

程序通过随机抽取的方式将Iris数据集分为训练集和测试集

用训练集训练后的贝叶斯分类器去分类测试集，多次测试后输出测试的平均精度

最后提供了输入接口，供用户输入一组数据，并输出其分类的结果

"""
import numpy as np
import pandas as pd
import math
import io
import sys
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

# 读取iris数据集
iris_csv = pd.read_csv(".\IrisDataset\iris.csv")
# 获得属性名
A = iris_csv.columns[1:-1]
# 载入数据集
iris_data =  iris_csv[iris_csv.columns[1:]]
iris_num = iris_csv[iris_csv.columns[0]]
Xor = iris_csv[A]
Yor = iris_csv[iris_csv.columns[-1]]
Xor = np.array(Xor)
Yor = np.array(Yor)
iris_num = range(len(iris_num))
# print(len(iris_num))
# 分类类别
C = np.unique(Yor)
# print(C)
# 参数
langta = 1 # 拉普拉斯平滑项
K = len(C)  # 类别总数
Na = len(A) # 属性数
# 训练集参数
Nk = [] # 训练集类别样本数
N = None  # 训练集个数
Ujk = {}    # 训练集均值
Sigmajk = {}   #训练集方差
train_ratio = 0.8 # 训练集比例
test_times = 10 # 测试次数


# 计算各类的属性的均值和方差
def cal_u_sigama_jk():
    # 计算各类的属性的均值和方差
    for j in range(Na):
        # 获取训练属性
        Xj = X[:,j]
        # print(Xj)
        Uk  =   []
        Sigmak  =   []
        for k in range(K):
            Xjk = Xj[Y == C[k]]
            ujk = sum(Xjk) / Nk[k]
            sigamajk = Xjk -ujk
            sigamajk = sum(np.square(sigamajk)) / Nk[k]
            Uk.append(ujk)
            Sigmak.append(sigamajk)
            # print(Xjk)
            # print(ujk)
            # print(sigamajk)
        Ujk[A[j]] = Uk
        Sigmajk[A[j]] = Sigmak
    pass


# 计算中间参数
def cal_mid_var():
    # cal_Nk
    for ck in C:
        Nk.append(sum(Y == ck))
    # cal_ujk
    cal_u_sigama_jk()

# 计算先验概率p(y = ck)
def cal_p_y_ck(Y,C,langta):
    k = 0
    P_ck = []
    for ck in C:
        # python list
        p =  Nk[k] + langta
        k = k + 1
        assert(N+K>0)
        p = p / (N + K*langta) 
        P_ck.append(p)
    return P_ck


# 计算类别条件概率密度
def cal_p_Xj_ajl_y_ck(aj,ajl):
    P_aj_ck = []
    # 计算属于各个类别的条件概率
    for k in range(K):
        sigma = Sigmajk[aj][k]
        u = Ujk[aj][k]
        P_u = -np.square(ajl - u)/ (2*np.square(sigma))
        P_u = math.exp(P_u)
        P_d = math.sqrt(2*math.pi)*sigma
        P = P_u/P_d
        P_aj_ck.append(P)
    return P_aj_ck

# 决策函数
def cal_loss_p_ck(xinput):
    Y_output = C[0]
    Pk = np.array([1,1,1])
    for j in range(Na):
        p_Xj_ajl_y_ck = cal_p_Xj_ajl_y_ck(A[j],xinput[j])
        p_Xj_ajl_y_ck = np.array(p_Xj_ajl_y_ck)
        Pk = np.multiply(Pk,p_Xj_ajl_y_ck)
    pass
    Y_index = np.argmax(Pk)
    Y_output = C[Y_index]
    return [Y_output,Y_index]
# 验证集合
def test_classify(X_test):
    Y_tag = []
    for xi in X_test:
        [Youtput,Y_index] = cal_loss_p_ck(xi)
        Y_tag.append(Youtput)
    count = 0
    for n in range(len(Y_test)):
        if Y_tag[n] == Y_test[n]:
            count = count+1
    accuracy = count/len(Y_test)
    # print(Ytest)
    print(accuracy)
    return accuracy

# 多次测试
print("=================================================")
print("Begin Test:")
Aver_accuracy = 0
for i in range(test_times):
    print("Test "+ str(i)+":")
    # 获得随机抽取索引
    iris_train_num = np.random.choice(iris_num, round(train_ratio*len(iris_num)), replace=False)
    iris_test_num = np.setdiff1d(iris_num,iris_train_num)
    # 训练集
    X = np.array(Xor[iris_train_num,])
    Y = np.array(Yor[iris_train_num])
    # 测试集
    X_test =  np.array(Xor[iris_test_num,])
    Y_test =  np.array(Yor[iris_test_num])
    Nk = [] # 训练集类别样本数
    N = len(Y)  # 训练集个数
    Ujk = {}    # 训练集均值
    Sigmajk = {}   #训练集方差
    cal_mid_var()
    # 计算中间变量
    # 计算先验概率p(y = ck)
    P_ck  = cal_p_y_ck(Y,C,langta)
    # 进行决策分类
    test_accuracy =test_classify(X_test)
    Aver_accuracy = Aver_accuracy + test_accuracy
    pass
Aver_accuracy = Aver_accuracy/test_times
print("End Test")
print("=================================================")
print("Aver accuracy: "+ str(Aver_accuracy))




print("**************************************************")
print("Exmaple:");
print("---------------------------------------------------")
print("Input:");
print("| Sepal.Length | Sepal.Width | Petal.Length | Petal.Width |\n")
print("    5.1           3.5        1.4          0.2")
print("---------------------------------------------------")
print("Output:");
print(" | Species | \n");
print("   setosa  ")
print("**************************************************")

print("=================================================")
print("Please Input you Test Data As Follow:\n");
print("| Sepal.Length | Sepal.Width | Petal.Length | Petal.Width |\n")
# 输入测试数据
x_test_data  =  [float(n) for n in input().split()]
lenth_input = len(x_test_data)
while lenth_input != 4:
    print("Please Re Input you Test Data As Follow:\n");
    print("| Sepal.Length | Sepal.Width | Petal.Length | Petal.Width |\n")
    x_test_data  =  [float(n) for n in input().split()]
    lenth_input = len(x_test_data)
[y_test_data,y_test_index] = cal_loss_p_ck(x_test_data)
print("")
print("Output:\n")
print(" | Species | \n");
print("  ",y_test_data) 
print("")
