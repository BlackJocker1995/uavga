class MLConfig:
    class ConstError(PermissionError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value

mlConfig = MLConfig()

# SITL Type PX4 or Ardupilot
mlConfig.MODE = 'Ardupilot'
# Output Debug information
mlConfig.DEBUG = False
# LSTM Input Length
mlConfig.INPUT_LEN = 3
# state + sensor
mlConfig.CONTEXT_LEN = 12
# size of input
mlConfig.DATA_LEN = mlConfig.CONTEXT_LEN + 20
# size of output
mlConfig.OUTPUT_DATA_LEN = 6
# Value retrans
mlConfig.RETRANS = True