from Cptool.config import toolConfig
from Cptool.gaMavlink import GaMavlinkAPM

if __name__ == '__main__':
    GaMavlinkAPM.extract_log_path(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_ga", threat=6)
