import geatpy as ea
import numpy as np

from Cptool.gaMavlink import GaMavlink
from uavga.uavgeat import UAVProblem


class UAVGA(object):
    def __init__(self, param_choice):

        para_dict = GaMavlink.load_param()
        # 获得sub 数据
        param_choice_dict = GaMavlink.select_sub_dict(para_dict, param_choice)
        step_unit = GaMavlink.read_unit_from_dict(param_choice_dict)

        self.param_len = len(param_choice)
        self.uavproblem = UAVProblem(param_choice, para_dict)
        self.NDSet = None
        self.population = None
        self.algorithm = None

        # 类型需要手动转换
        self.default_pop = (GaMavlink.get_default_values(param_choice_dict) / step_unit).to_numpy(dtype=int)

    def set_model(self, filename):
        self.uavproblem.set_model(filename)

    def set_trans(self, filename):
        self.uavproblem.set_trans(filename)

    def run(self):
        NINDs = 1000
        Encoding = 'RI'  # 编码方式
        Field = ea.crtfld(Encoding, self.uavproblem.varTypes, self.uavproblem.ranges,
                          self.uavproblem.borders)  # 创建区域描述器
        population = (ea.Population(Encoding, Field, NINDs))  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        # 自定义初始化的种群soea_DE_currentToBest_1_bin_templet
        """===============================算法参数设置============================="""
        self.algorithm = ea.soea_DE_currentToBest_1_bin_templet(self.uavproblem, population)  # 实例化一个算法模板对象
        self.algorithm.MAXGEN = 50  # 最大进化代数
        self.algorithm.mutOper.F = 0.7  # 差分进化中的参数F
        self.algorithm.recOper.XOVR = 0.7  # 重组概率
        self.algorithm.trappedValue = 0.1  # “进化停滞”判断阈值
        self.algorithm.maxTrappedCount = 10  # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
        self.algorithm.drawing = 0  #
        """===========================根据先验知识创建先知种群======================="""
        prophetChrom = self.default_pop  # 假设已知为一条比较优秀的染色体
        prophetPop = ea.Population(Encoding, Field, 1, prophetChrom)  # 实例化种群对象（设置个体数为1）

        self.algorithm.call_aimFunc(prophetPop)
        """==========================调用算法模板进行种群进化======================="""
        [self.NDSet, self.population] = self.algorithm.run(prophetPop)

    def return_best_n_gen(self, n=1):
        if (self.BestIndi is None) or (self.population is None):
            raise ValueError('Please run() at first')

        obj_trace = self.population.Phen
        var_trace = self.population.ObjV

        obj_trace = np.array(obj_trace)
        var_trace = np.array(var_trace)

        # 去除重复
        candidate_var_index = np.unique(var_trace, axis=0, return_index=True)[1]
        candidate_var = var_trace[candidate_var_index].reshape(-1)
        candidate_obj = obj_trace[candidate_var_index]

        candidate = self.uavproblem.maxormins * candidate_var
        # 从小到大
        candidate_index = np.argsort(candidate)
        candidate_obj = candidate_obj[candidate_index]

        return self.uavproblem.reasonable_range(candidate_obj)[:n]


