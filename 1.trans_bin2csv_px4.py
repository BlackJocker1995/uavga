from Cptool.config import toolConfig
from Cptool.gaMavlink import GaMavlinkPX4

if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    GaMavlinkPX4.extract_log_path(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/ulg_changed")
