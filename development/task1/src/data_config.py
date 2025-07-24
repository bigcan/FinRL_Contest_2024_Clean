import os

class ConfigData:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Auto-detect absolute path to data directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..')
            data_dir = os.path.join(project_root, 'data', 'raw', 'task1')
        
        self.data_dir = data_dir
        self.csv_path = f"{data_dir}/BTC_1sec.csv"
        self.predict_ary_path = f"{data_dir}/BTC_1sec_predict.npy"