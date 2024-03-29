# -*- coding: utf-8 -*-
# @Time    : 2020-12-27 11:14
# @Author  : Feilin Zhu--Hohai University
# @File    : main.py
# 机器学习预测地下水水位模型（进化算法超参数优化 + K-fold交叉验证 + 并行计算）


import geatpy as ea  # import geatpy
from MyProblem import MyProblem  # 导入自定义问题接口

"""
程序说明：该程序利用进化算法 + K-fold交叉验证 + 并行计算 来优化GBDT模型中的两个参数：learning_rate 和 n_estimators
在执行脚本main.py中设置PoolType字符串来控制采用的是多进程还是多线程。
注意：使用多进程时，程序必须以“if __name__ == '__main__':”作为入口，
      这个是multiprocessing的多进程模块的硬性要求。
"""

if __name__ == '__main__':
    """===============================实例化问题对象==========================="""
    PoolType = 'Process'  # 设置采用多进程，若修改为: PoolType = 'Thread'，则表示用多线程
    problem = MyProblem(PoolType)  # 生成问题对象

    """=================================种群设置=============================="""
    Encoding = 'RI'  # 编码方式
    NIND = 20  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）

    """===============================算法参数设置============================="""
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)  # 实例化一个算法模板对象
    # myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 100  # 最大进化代数
    # myAlgorithm.trappedValue = 1e-6  # “进化停滞”判断阈值
    # myAlgorithm.maxTrappedCount = 20  # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
    myAlgorithm.logTras = 1  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）

    """==========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    BestIndi.save()  # 把最优个体的信息保存到文件中

    """=================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最优的目标函数值为：%s' % (BestIndi.ObjV[0][0]))
        print('最优的超参数值为：')
        for i in range(BestIndi.Phen.shape[1]):
            print(BestIndi.Phen[0, i])
    else:
        print('没找到可行解。')
