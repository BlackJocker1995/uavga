import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt
import pickle

from Cptool.config import toolConfig
from Cptool.gaMavlink import GaMavlinkAPM
from Cptool.mavtool import load_param, select_sub_dict, read_unit_from_dict, get_default_values, read_range_from_dict
from range.rangeproblem import RangeProblem, GARangeProblem


class RangeOptimizer(object):
    def __init__(self):
        self.problem = RangeProblem()
        self.start_value = None
        # Parameter
        self.participle_param = toolConfig.PARAM
        para_dict = load_param()

        # default value, step and boundary
        self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)
        self.step_unit = read_unit_from_dict(self.param_choice_dict)
        self.default_pop = get_default_values(self.param_choice_dict)
        self.sub_value_range = read_range_from_dict(self.param_choice_dict)

    def set_bounds(self):
        # step
        self.problem.init_bounds_and_step(self.sub_value_range, self.step_unit)

class GARangeOptimizer(RangeOptimizer):
    def __init__(self, result_data):
        super(GARangeOptimizer, self).__init__()

        name = 'ANAProblem'  # 初始化name（函数名称，可以随意设置）boundary
        M = 2  # 初始化M（目标维数）
        maxormins = [-1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = self.sub_value_range.shape[0] * 2  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = np.repeat(self.sub_value_range[:, 0] / self.step_unit, 2)  # 决策变量下界
        lb[1::2] = self.default_pop // self.step_unit
        ub = np.repeat(self.sub_value_range[:, 1] / self.step_unit, 2)  # 决策变量上界
        ub[::2] = self.default_pop // self.step_unit
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化

        self.problem = GARangeProblem(name=name, M=M, maxormins=maxormins, Dim=self.sub_value_range.shape[0],
                                 varTypes=varTypes, lb=lb, ub=ub, lbin=lbin, ubin=ubin, result_data=result_data)

        # Result logging
        self.NDSet = None
        self.population = None
        self.algorithm = None

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
        [NDSet, population] = self.algorithm.run()

        with open('NDSetnew.pkl', 'wb') as f:
            pickle.dump(NDSet, f)
        NDSet.save()  # 把非支配种群的信息保存到文件中

        ea.moeaplot(NDSet.ObjV, xyzLabel=['No. of Validated Configuration', 'Incorrect/Validated Ratio'])

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


