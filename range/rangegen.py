import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt
import pickle

from Cptool.gaMavlink import GaMavlinkAPM
from range.rangegeat import ANAProblem

class ANAGA(object):
    def __init__(self, param_choice, result_data):
        para_dict = GaMavlinkAPM.load_param()
        # 获得sub 数据
        param_choice_dict = GaMavlinkAPM.select_sub_dict(para_dict, param_choice)
        step_unit = GaMavlinkAPM.read_unit_from_dict(param_choice_dict)

        self.param_len = len(param_choice)
        self.problem = ANAProblem(param_choice, para_dict, result_data)
        self.algorithm = None
        self.obj_trace = None
        self.var_trace = None

        self.default_pop = (GaMavlinkAPM.get_default_values(param_choice_dict) / step_unit).to_numpy(dtype=int)

    def run(self):
        NINDs = 3000
        Encoding = 'RI'  # 编码方式
        Field = ea.crtfld(Encoding, self.problem.varTypes, self.problem.ranges,
                          self.problem.borders)  # 创建区域描述器
        population = ea.Population(Encoding, Field, NINDs) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        # 自定义初始化的种群 moea_NSGA2_templet
        """===============================算法参数设置============================="""
        self.algorithm = ea.moea_NSGA2_templet(self.problem, population)  # 实例化一个算法模板对象
        self.algorithm.MAXGEN = 300 # 最大进化代数
        self.algorithm.mutOper.Pm = 0.5  # 修改变异算子的变异概率
        self.algorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
        self.algorithm.maxTrappedCount = 10
        self.algorithm.drawing = 1#
        """==========================调用算法模板进行种群进化======================="""
        [NDSet, population]  = self.algorithm.run()

        with open('NDSetnew.pkl','wb') as f:
            pickle.dump(NDSet, f)
        NDSet.save()  # 把非支配种群的信息保存到文件中

        ea.moeaplot(NDSet.ObjV, xyzLabel=['No. of Solutions Covered by Range', 'Safe/Pass Ratio of Covered Solutions'])

        # 输出
        print('用时：%s 秒' % (self.algorithm.passTime))
        print('非支配个体数：%s 个' % (NDSet.sizes))
        print('单位时间找到帕累托前沿点个数：%s 个' % (int(NDSet.sizes // self.algorithm.passTime)))

        # 计算指标
        PF = self.problem.getReferObjV()  # 获取真实前沿，详见Problem.py中关于Problem类的定义
        if PF is not None and NDSet.sizes != 0:
            GD = ea.indicator.GD(NDSet.ObjV, PF)  # 计算GD指标
            IGD = ea.indicator.IGD(NDSet.ObjV, PF)  # 计算IGD指标
            HV = ea.indicator.HV(NDSet.ObjV, PF)  # 计算HV指标
            Spacing = ea.indicator.Spacing(NDSet.ObjV)  # 计算Spacing指标
            print('GD', GD)
            print('IGD', IGD)
            print('HV', HV)
            print('Spacing', Spacing)
        """============================进化过程指标追踪分析==========================="""
        if PF is not None:
            metricName = [['IGD'], ['HV']]
            [NDSet_trace, Metrics] = ea.indicator.moea_tracking(self.algorithm.pop_trace, PF, metricName,
                                                                self.problem.maxormins)
            # 绘制指标追踪分析图
            ea.trcplot(Metrics, labels=metricName, titles=metricName)

    def return_best_n_gen(self, n=1):
        if (self.algorithm is None) or (self.obj_trace is None) or (self.var_trace is None):
            raise ValueError('Please run() at first')

        candidate = self.problem.maxormins * self.obj_trace[:, 1]

        if n == 0:
            top_gen = np.zeros(len(candidate), dtype=int)
            for i in range(len(candidate)):
                min_index = np.argmin(candidate)
                top_gen[i] = min_index  # 记录最优种群个体是在哪一代
                candidate[min_index] = 0
        else:
            top_gen = np.zeros(n, dtype=int)
            for i in range(n):
                min_index = np.argmin(candidate)
                top_gen[i] = min_index  # 记录最优种群个体是在哪一代
                candidate[min_index] = 0
        params = self.var_trace[top_gen, :]
        return self.problem.reasonable_range(params)


