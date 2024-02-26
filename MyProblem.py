# -*- coding: utf-8 -*-
# @Time    : 2020-12-27 11:14
# @Author  : Feilin Zhu--Hohai University
# @File    : main.py
# 机器学习预测地下水水位模型（进化算法超参数优化+K-fold交叉验证+并行计算）


import numpy as np
import geatpy as ea
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from sklearn import ensemble

"""
程序说明：该程序利用进化算法 + K-fold交叉验证 + 并行计算 来优化GBDT模型中的三个个参数：learning_rate、n_estimators、max_depth
在执行脚本main.py中设置PoolType字符串来控制采用的是多进程还是多线程。
注意：使用多进程时，程序必须以“if __name__ == '__main__':”作为入口，
      这个是multiprocessing的多进程模块的硬性要求。
"""

# 基本参数设置
jinghao = '#070107'  # 井号
chidu = '月'  # 时间尺度
yujianqi = '1-step'  # 预见期（步数）
H = int(yujianqi[0])  # 预见期
lags = [3, 0, 1, 0, 0, 0, 0, 1, 0]  # 输入因子的时滞，0表示不考虑该因子
percent = 0.7  # 前百分之多少样本用于训练，剩余样本用于模型测试


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 3  # 初始化Dim（决策变量维数，超参数有2个）
        varTypes = [0, 1, 1]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0.01, 50, 1]  # 决策变量下界
        ub = [0.50, 500, 10]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        # 导入训练阶段计算目标函数要用到的一些数据
        data = np.loadtxt('./输入数据/' + jinghao + '_训练样本_' + chidu + '.csv', delimiter=',', skiprows=1,
                          encoding='utf-8')  # 指定读取的列

        Num = len(data)  # 获取样本总数
        index = round(Num * percent)  # index表示训练数据最后一行的行号，用于分割训练集和测试集

        # 根据预见期和各因子的时滞，重构训练样本
        drop_num = H + max(lags) - 1  # 丢弃样本的个数

        # 输出因子整理
        y_data = data[drop_num:, 0]

        # 输入因子整理
        x_data = np.zeros((Num - drop_num, sum(lags)))

        col_new = 0  # 新x_data数组的列序号
        col_old = 0  # 原始data数组的列序号
        for lag in lags:
            if lag > 1:
                for j in range(lag):
                    x_data[:, col_new + j] = data[(max(lags) - 1 - j):(max(lags) - 1 - j + Num - drop_num), col_old]
                col_new += lag

            elif lag == 1:
                x_data[:, col_new] = data[max(lags) - 1:(max(lags) - 1 + Num - drop_num), col_old]
                col_new += 1

            col_old += 1

        # 分割训练样本和测试样本
        x_train, x_test = x_data[0:index, :], x_data[index:, :]
        y_train, y_test = y_data[0:index], y_data[index:]

        # 输入因子数据标准化处理
        min_max_scaler = preprocessing.MinMaxScaler((0, 1))  # 将输入因子缩放到指定范围，通常在零和一之间
        self.x_train = min_max_scaler.fit_transform(x_train)  # 训练集数据标准化
        self.y_train = y_train

        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(8)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)), [Vars] * pop.sizes, [self.x_train] * pop.sizes, [self.y_train] * pop.sizes))
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            pop.ObjV = np.array(result.get())


def subAimFunc(args):
    i = args[0]  # 第i个个体
    Vars = args[1]  # 所有个体的决策变量矩阵
    x_train = args[2]  # 第i个个体的x_train
    y_train = args[3]  # 第i个个体的y_train
    L_rate = Vars[i, 0]  # 获取第i个个体的学习率超参数
    n_est = int(Vars[i, 1])  # 获取第i个个体的迭代次数超参数，注意：要转换成整数类型，否则报错
    max_dep = int(Vars[i, 2])  # 获取第i个个体的决策树最大深度超参数，注意：要转换成整数类型，否则报错
    GBDT = ensemble.GradientBoostingRegressor(learning_rate=L_rate, n_estimators=n_est, max_depth=max_dep).fit(x_train,
                                                                                                               y_train)  # 创建指定超参数值的GBDT模型

    # 采用k-折交叉验证，可减少过拟合，但计算量翻k倍
    # 评价指标：r2、explained_variance、neg_mean_absolute_error、neg_mean_squared_error，均为越大越优
    scores = cross_val_score(GBDT, x_train, y_train, cv=10, scoring='r2')  # 计算交叉验证的得分，评价指标为r2

    ObjV_i = [scores.mean()]  # 把交叉验证的平均得分作为目标函数值
    return ObjV_i
