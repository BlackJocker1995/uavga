import Cptool.config
from Cptool.gaMavlink import GaMavlink

if __name__ == '__main__':
    GaMavlink.extract_from_log_path(f"./log/{Cptool.config.MODE}")
