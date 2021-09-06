from Cptool.gaMavlink import GaMavlink
import Cptool.config

if __name__ == '__main__':
    GaMavlink.extract_from_log_path(f"./log/{Cptool.config.MODE}")
