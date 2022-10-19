from Cptool.config import toolConfig
import pandas as pd

from range.rangesearcher import GARangeOptimizer

if __name__ == '__main__':
    # The parameters you want to fuzzing, they must be corresponding to the predictor had.
    toolConfig.select_mode("PX4")

    result_data = pd.read_csv(f'result/{toolConfig.MODE}/params{toolConfig.EXE}.csv', header=0).drop(columns="score")
    ga = GARangeOptimizer(result_data)
    ga.set_bounds()
    ga.run()