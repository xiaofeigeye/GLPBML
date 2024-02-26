# -*- coding: utf-8 -*-
# @Time    : 2020-11-23 11:14
# @Author  : Feilin Zhu--Hohai University
# @File    : 机器学习预测模型.py


import pandas as pd
import numpy as np
import time
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pyecharts import Line
import math

# ==========导入、整理训练样本数据==========

# 基本参数设置
jinghao = '#070107'  # 井号
chidu = '月'  # 时间尺度
yujianqi = '1-step'  # 预见期（步数）
H = int(yujianqi[0])  # 预见期
lags = [3, 0, 1, 0, 0, 0, 0, 1, 0]  # 输入因子的时滞，0表示不考虑该因子
percent = 0.7  # 前百分之多少样本用于训练，剩余样本用于模型测试

# ==========导入优化后的超参数==========
canshu = np.loadtxt('./Result/Phen.csv', delimiter=',')
L_rate = canshu[0]  # 优化后的学习率参数 learning_rate
n_est = int(canshu[1])  # 优化后的迭代次数参数 n_estimators
max_dep = int(canshu[2])  # 优化后的决策树最大深度参数 max_depth

# 读取数据文件
input = pd.read_excel('./输入数据/' + jinghao + '_训练样本_' + chidu + '.xlsx', sheet_name=0, header=0, skiprows=None,
                      index_col=0)

data = np.asarray(input)  # DataFrame转ndarray
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

n = x_train.shape[1]  # 获取输入变量数

# 输入因子标准化处理
min_max_scaler = preprocessing.MinMaxScaler((0, 1))  # 将输入因子缩放到指定范围，通常在零和一之间
x_train = min_max_scaler.fit_transform(x_train)  # 训练集数据标准化
x_test = min_max_scaler.transform(x_test)  # 在新的不可预见的测试集上实施和训练集相同的缩放和移位操作


# ==========运行机器学习模型、输出结果==========
def run_DL(model, Loc='#000111', chidu='五日', H='1-step'):
    start = time.time()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)  # 计算score值

    prediction_train = model.predict(x_train)  # 模拟（训练期）
    mae_train = mean_absolute_error(y_train, prediction_train)  # 平均绝对误差（训练期）
    rmse_train = math.sqrt(mean_squared_error(y_train, prediction_train))  # 均方根误差（训练期）
    ev_train = explained_variance_score(y_train, prediction_train)  # 可释方差值（训练期），反映自变量与因变量之间的相关程度，最好为1，越低越差
    corr_train = np.corrcoef(y_train, prediction_train)[0, 1]  # 皮尔逊线性相关系数（训练期），函数返回的是相关系数矩阵，所以加[0, 1]
    r2_train = r2_score(y_train, prediction_train)  # 确定性系数（训练期）

    prediction_test = model.predict(x_test)  # 外延（验证期）
    mae_test = mean_absolute_error(y_test, prediction_test)  # 平均绝对误差（验证期）
    rmse_test = math.sqrt(mean_squared_error(y_test, prediction_test))  # 均方根误差（验证期）
    ev_test = explained_variance_score(y_test, prediction_test)  # 可释方差值（验证期）
    corr_test = np.corrcoef(y_test, prediction_test)[0, 1]  # 皮尔逊线性相关系数（验证期）
    r2_test = r2_score(y_test, prediction_test)  # 确定性系数（验证期）

    end = time.time()

    print("=====================================================================")
    print("训练期：平均绝对误差(MAE) = %.4f, 均方根误差(RMSE) = %.4f, 可释方差(EV) = %.4f, 线性相关系数(R) = %.4f, 确定性系数(NSE) = %.4f." % (
        mae_train, rmse_train, ev_train, corr_train, r2_train))
    print("验证期：平均绝对误差(MAE) = %.4f, 均方根误差(RMSE) = %.4f, 可释方差(EV) = %.4f, 线性相关系数(R) = %.4f, 确定性系数(NSE) = %.4f." % (
        mae_test, rmse_test, ev_test, corr_test, r2_test))
    print("计算耗时： %.2f 秒." % (end - start))
    print("====================================================================")

    # 结果绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    x1 = np.arange(1, index + 1)
    x2 = np.arange(index + 1, Num + 1 - drop_num)
    plt.figure(figsize=(12, 6))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(x1, prediction_train, 'b-', label="模型预测(训练期)", linewidth=1.5)
    plt.plot(x1, y_train, 'r-', label="实测数据(训练期)", linewidth=1.5)
    plt.plot(x2, prediction_test, 'k-', label="模型预测(检验期)", linewidth=1.5)
    plt.plot(x2, y_test, 'y-', label="实测数据(检验期)", linewidth=1.5)
    plt.axvline(x=index, ls="--", c="fuchsia", linewidth=2)  # 添加训练期和检验期的分割线

    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.xlabel("时序(" + chidu + ')', fontsize=14, fontweight='bold')
    plt.ylabel("地下水埋深(m)", fontsize=14, fontweight='bold')
    plt.legend(loc=0)  # loc图例位置，numpoints图例中marker的个数
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=14, fontweight='bold')  # 设置图例字体的大小和粗细
    plt.xlim(0, Num + 1)  # 设置x轴的范围
    plt.tight_layout()  # 紧凑布局
    plt.savefig('./输出结果/' + Loc + '_' + chidu + '_' + H + '_' + '训练期和验证期结果' + '.jpg', dpi=300)
    plt.show()

    # 单独绘制验证期结果
    plt.figure()
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    plt.plot(x2, prediction_test, 'b-o', label='模型预测(检验期)', markersize=3)
    plt.plot(x2, y_test, 'r-^', label='实测数据(检验期)', markersize=3)
    plt.title('score: %f' % score)
    plt.xlabel("时序(" + chidu + ')', fontweight='bold')
    plt.ylabel("地下水埋深(m)", fontweight='bold')
    plt.xlim(index + 1, Num + 1)  # 设置x轴的范围
    plt.tight_layout()  # 紧凑布局
    plt.legend(loc=0, numpoints=2)  # loc图例位置，numpoints图例中marker的个数
    plt.savefig('./输出结果/' + Loc + '_' + chidu + '_' + H + '_' + '验证期结果' + '.jpg', dpi=300)
    plt.show()

    # 绘制实测与预测结果散点图
    plt.figure()
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    plt.scatter(y_train, prediction_train, c='b', marker='o', alpha=0.5, label='训练期')
    plt.scatter(y_test, prediction_test, c='r', marker='^', alpha=0.5, label='检验期')
    plt.xlabel('实测值(m)', fontsize=14, fontweight='bold')
    plt.ylabel('预测值(m)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.tight_layout()  # 紧凑布局
    plt.legend(loc=0)  # loc图例位置
    plt.savefig('./输出结果/' + Loc + '_' + chidu + '_' + H + '_' + '实测-预测散点图' + '.jpg', dpi=300)
    plt.show()

    # Pyechart绘制训练和验证期结果
    line = Line("训练期和验证期模拟结果")
    xx = np.hstack([x1, x2])
    yy1 = np.hstack([prediction_train, prediction_test])  # 模型模拟序列合并
    yy2 = np.hstack([y_train, y_test])  # 实测序列合并
    line.add('模型模拟', xx, yy1, line_color='red', is_smooth=False, is_datazoom_show=True)
    line.add('实测数据', xx, yy2, line_color='blue', is_smooth=False, is_datazoom_show=True, mark_line=['max', 'min'])
    line.render('./输出结果/' + Loc + '_' + chidu + '_' + H + '_' + '训练期和验证期模拟结果' + '.html')

    # 结果导出Excel文件
    shixu = [i for i in range(1, Num + 1 - drop_num)]
    biaotou = ["地下水埋深(实测值)", "地下水埋深(模型预测)", "绝对误差(m)", "相对误差(%)"]
    zonghe1 = np.vstack((y_train, prediction_train))
    zonghe2 = np.vstack((y_test, prediction_test))
    zonghe3 = np.hstack((zonghe1, zonghe2)).T
    jueduiwucha = zonghe3[:, [1]] - zonghe3[:, [0]]  # 计算绝对误差
    xiangduiwucha = (zonghe3[:, [1]] - zonghe3[:, [0]]) / zonghe3[:, [0]] * 100  # 计算相对误差
    zonghe4 = np.hstack((zonghe3, jueduiwucha, xiangduiwucha))
    input_new = input.iloc[drop_num:, :]  # 删除前几行，构造新序列的index，用于结果输出时保存日期
    zonghe_out = round(pd.DataFrame(zonghe4, index=input_new.index, columns=biaotou), 3)
    zonghe_out.insert(0, "时序", shixu)

    row_name = ['平均绝对误差', '均方根误差', '可释方差', '线性相关系数', '确定性系数']
    col_name = ['训练期', '验证期']
    zhibiao_train = np.vstack((mae_train, rmse_train, ev_train, corr_train, r2_train))
    zhibiao_test = np.vstack((mae_test, rmse_test, ev_test, corr_test, r2_test))
    zhibiao = round(pd.DataFrame(np.hstack((zhibiao_train, zhibiao_test)), index=row_name, columns=col_name), 4)

    writer = pd.ExcelWriter('./输出结果/' + Loc + '_' + chidu + '_' + H + '_' + '预测结果' + '.xlsx')
    zonghe_out.to_excel(writer, sheet_name="预测结果")
    zhibiao.to_excel(writer, sheet_name="评价指标")
    writer.close()


from sklearn.ensemble import GradientBoostingRegressor

model_GBDT = GradientBoostingRegressor(learning_rate=L_rate, n_estimators=n_est, max_depth=max_dep)  # 采用三个超参数优化值
# model_GBDT = GradientBoostingRegressor()  # 采用超参数默认值
# model_GBDT = GradientBoostingRegressor(learning_rate=0.2, n_estimators=150, max_depth=5)  # 采用三个超参数优化值

if __name__ == "__main__":
    run_DL(model_GBDT, Loc=jinghao, chidu=chidu, H=yujianqi)
