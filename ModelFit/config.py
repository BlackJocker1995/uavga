# SITL类型 PX4 和 Ardupilot
MODE = 'Ardupilot'

# 是否输出Debug信息
DEBUG = False

# LSTM的输入长度
INPUT_LEN = 3

#
CONTEXT_LEN = 12

# 每一个input数据的长度
DATA_LEN = CONTEXT_LEN + 20

# 输出的数据长度
OUTPUT_DATA_LEN = 6

# 是否还原
RETRANS = True