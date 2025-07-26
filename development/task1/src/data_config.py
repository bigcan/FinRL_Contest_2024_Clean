import os
import json

class ConfigData:
    def __init__(self, data_dir: str = None, config_file: str = None):
        # Load configuration from JSON if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            data_paths = config.get('data_paths', {})
            self.csv_path = data_paths.get('csv_path', 'data/BTC_1sec.csv')
            self.predict_ary_path = data_paths.get('predict_path', 'data/BTC_1sec_predict.npy')
        else:
            # Fallback to default behavior
            if data_dir is None:
                # Auto-detect absolute path to data directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.join(current_dir, '..', '..', '..')
                data_dir = os.path.join(project_root, 'data', 'raw', 'task1')
            
            self.data_dir = data_dir
            self.csv_path = f"{data_dir}/BTC_1sec.csv"
            self.predict_ary_path = f"{data_dir}/BTC_1sec_predict.npy"