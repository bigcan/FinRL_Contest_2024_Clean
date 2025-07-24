#!/usr/bin/env python3
"""
FinRL Contest 2024 - Project Setup Script

This script initializes the complete file organization system and validates the setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
from datetime import datetime
import argparse

class ProjectSetup:
    """Project setup and initialization manager."""
    
    def __init__(self, project_root: str = None):
        """Initialize project setup."""
        if project_root is None:
            project_root = Path(__file__).parent
        self.project_root = Path(project_root).resolve()
        self.setup_log = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log setup messages."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.setup_log.append(log_entry)
        print(log_entry)
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        self.log("Checking prerequisites...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.log(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}", "ERROR")
            return False
        
        self.log(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check if original directories exist
        original_dirs = ["original/Task_1_starter_kit", "original/Task_2_starter_kit"]
        for dir_path in original_dirs:
            if not (self.project_root / dir_path).exists():
                self.log(f"Missing original directory: {dir_path}", "ERROR")
                return False
        
        self.log("✅ Original directories found")
        
        # Check if data directories are set up
        data_dirs = ["data/raw/task1", "data/raw/task2"]
        for dir_path in data_dirs:
            if not (self.project_root / dir_path).exists():
                self.log(f"Creating data directory: {dir_path}")
                (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        return True
    
    def setup_environments(self) -> bool:
        """Set up virtual environments for both tasks."""
        self.log("Setting up virtual environments...")
        
        try:
            # Task 1 environment
            task1_env = self.project_root / "development" / "environments" / "task1_env"
            if not task1_env.exists():
                self.log("Creating Task 1 virtual environment...")
                subprocess.run([sys.executable, "-m", "venv", str(task1_env)], check=True)
                
                # Install Task 1 dependencies
                pip_path = task1_env / "bin" / "pip" if os.name != "nt" else task1_env / "Scripts" / "pip.exe"
                requirements_path = self.project_root / "development" / "task1" / "src" / "requirements_simplified.txt"
                
                if requirements_path.exists():
                    subprocess.run([str(pip_path), "install", "-r", str(requirements_path)], check=True)
                    self.log("✅ Task 1 dependencies installed")
                else:
                    self.log("⚠️ Task 1 requirements file not found", "WARNING")
            else:
                self.log("✅ Task 1 environment already exists")
            
            # Task 2 environment
            task2_env = self.project_root / "development" / "environments" / "task2_env"
            if not task2_env.exists():
                self.log("Creating Task 2 virtual environment...")
                subprocess.run([sys.executable, "-m", "venv", str(task2_env)], check=True)
                
                # Install Task 2 dependencies
                pip_path = task2_env / "bin" / "pip" if os.name != "nt" else task2_env / "Scripts" / "pip.exe"
                requirements_path = self.project_root / "development" / "task2" / "src" / "requirements_simplified.txt"
                
                if requirements_path.exists():
                    subprocess.run([str(pip_path), "install", "-r", str(requirements_path)], check=True)
                    self.log("✅ Task 2 dependencies installed")
                else:
                    self.log("⚠️ Task 2 requirements file not found", "WARNING")
            else:
                self.log("✅ Task 2 environment already exists")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Error setting up environments: {e}", "ERROR")
            return False
    
    def create_config_templates(self) -> bool:
        """Create configuration templates for experiments."""
        self.log("Creating configuration templates...")
        
        try:
            # Task 1 baseline config
            task1_config = {
                "experiment": {
                    "name": "task1_baseline",
                    "description": "Baseline ensemble experiment"
                },
                "model": {
                    "agents": ["D3QN", "DoubleDQN", "TwinD3QN"],
                    "ensemble_method": "voting",
                    "net_dims": [128, 128, 128]
                },
                "training": {
                    "max_episodes": 1000,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "replay_buffer_size": 100000
                },
                "environment": {
                    "initial_cash": 1000000,
                    "data_path": "data/raw/task1/BTC_1sec_predict.npy"
                },
                "logging": {
                    "level": "INFO",
                    "save_intermediate": True,
                    "log_dir": "debugging/logs/task1/"
                }
            }
            
            config_path = self.project_root / "development" / "task1" / "configs" / "baseline_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(task1_config, f, indent=2)
            
            self.log("✅ Task 1 baseline config created")
            
            # Task 2 baseline config
            task2_config = {
                "experiment": {
                    "name": "task2_baseline",
                    "description": "Baseline RLMF experiment"
                },
                "model": {
                    "name": "meta-llama/Llama-3.2-3B-Instruct",
                    "lora_r": 30,
                    "lora_alpha": 16,
                    "lora_dropout": 0.1,
                    "quantization": "4bit"
                },
                "training": {
                    "max_steps": 100,
                    "learning_rate": 1e-5,
                    "signal_strength": 10,
                    "start_date": "2022-10-10",
                    "end_date": "2022-10-31"
                },
                "environment": {
                    "lookahead": 3,
                    "num_long": 3,
                    "num_short": 3,
                    "tickers": ["AAPL", "NVDA", "GOOG", "AMZN", "MSFT", "XOM", "WMT"]
                },
                "logging": {
                    "level": "INFO",
                    "save_intermediate": True,
                    "log_dir": "debugging/logs/task2/"
                }
            }
            
            config_path = self.project_root / "development" / "task2" / "configs" / "baseline_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(task2_config, f, indent=2)
            
            self.log("✅ Task 2 baseline config created")
            
            return True
            
        except Exception as e:
            self.log(f"Error creating config templates: {e}", "ERROR")
            return False
    
    def create_example_scripts(self) -> bool:
        """Create example training and evaluation scripts."""
        self.log("Creating example scripts...")
        
        try:
            # Task 1 training script
            task1_train_script = '''#!/usr/bin/env python3
"""
Task 1 Training Script - Cryptocurrency Trading with Ensemble Learning
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "development" / "task1" / "src"))

def main():
    parser = argparse.ArgumentParser(description="Task 1 Training")
    parser.add_argument("--config", default="baseline_config.json", help="Configuration file")
    parser.add_argument("--output-dir", default="results/task1_results/", help="Output directory")
    parser.add_argument("--experiment-name", default="task1_baseline", help="Experiment name")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = project_root / "development" / "task1" / "configs" / args.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Starting Task 1 training with config: {args.config}")
    print(f"Experiment: {args.experiment_name}")
    
    # Import and run training
    try:
        from task1_ensemble import main as train_ensemble
        train_ensemble()
        print("✅ Training completed successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure Task 1 data is downloaded and environment is activated")

if __name__ == "__main__":
    main()
'''
            
            script_path = self.project_root / "development" / "task1" / "scripts" / "train.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(script_path, 'w') as f:
                f.write(task1_train_script)
            
            os.chmod(script_path, 0o755)
            self.log("✅ Task 1 training script created")
            
            # Task 2 training script
            task2_train_script = '''#!/usr/bin/env python3
"""
Task 2 Training Script - LLM-Engineered Signals with RLMF
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "development" / "task2" / "src"))

def main():
    parser = argparse.ArgumentParser(description="Task 2 Training")
    parser.add_argument("--config", default="baseline_config.json", help="Configuration file")
    parser.add_argument("--output-dir", default="results/task2_results/", help="Output directory")
    parser.add_argument("--experiment-name", default="task2_baseline", help="Experiment name")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = project_root / "development" / "task2" / "configs" / args.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Starting Task 2 training with config: {args.config}")
    print(f"Experiment: {args.experiment_name}")
    
    # Import and run training
    try:
        from task2_train import train
        train()
        print("✅ Training completed successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure HuggingFace access is set up and environment is activated")

if __name__ == "__main__":
    main()
'''
            
            script_path = self.project_root / "development" / "task2" / "scripts" / "train.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(script_path, 'w') as f:
                f.write(task2_train_script)
            
            os.chmod(script_path, 0o755)
            self.log("✅ Task 2 training script created")
            
            return True
            
        except Exception as e:
            self.log(f"Error creating example scripts: {e}", "ERROR")
            return False
    
    def run_validation_tests(self) -> bool:
        """Run validation tests to ensure setup is correct."""
        self.log("Running validation tests...")
        
        try:
            # Test imports
            sys.path.append(str(self.project_root / "testing" / "unit_tests"))
            from run_tests import TestRunner
            
            runner = TestRunner(str(self.project_root))
            
            # Run data validation only
            results = runner.run_data_validation()
            
            if results["status"] == "success":
                if results["issues"]:
                    self.log(f"⚠️ Data validation completed with {len(results['issues'])} issues", "WARNING")
                    for issue in results["issues"]:
                        self.log(f"  - {issue}", "WARNING")
                else:
                    self.log("✅ Data validation passed")
            else:
                self.log("❌ Data validation failed", "ERROR")
                return False
            
            return True
            
        except Exception as e:
            self.log(f"Error running validation tests: {e}", "ERROR")
            return False
    
    def generate_setup_report(self) -> str:
        """Generate setup completion report."""
        self.log("Generating setup report...")
        
        report_path = self.project_root / "documentation" / "setup_reports" / f"setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        html_content = [
            "<html><head><title>FinRL Contest 2024 - Setup Report</title></head><body>",
            "<h1>FinRL Contest 2024 - Project Setup Report</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "<h2>Setup Log</h2>",
            "<pre>"
        ]
        
        for log_entry in self.setup_log:
            html_content.append(log_entry)
        
        html_content.extend([
            "</pre>",
            "<h2>Next Steps</h2>",
            "<ol>",
            "<li><strong>Download Task 1 Data:</strong> Download Bitcoin LOB data from Google Drive to data/raw/task1/</li>",
            "<li><strong>Set up HuggingFace Access:</strong> Login to HuggingFace and accept Llama model license</li>",
            "<li><strong>Activate Environments:</strong> Use source development/environments/task1_env/bin/activate</li>",
            "<li><strong>Run Training:</strong> Use development/task1/scripts/train.py or development/task2/scripts/train.py</li>",
            "<li><strong>Run Tests:</strong> Use testing/unit_tests/run_tests.py --all</li>",
            "</ol>",
            "<h2>Directory Structure</h2>",
            "<p>The complete file organization system has been set up with the following structure:</p>",
            "<ul>",
            "<li>📁 original/ - Preserved original files</li>",
            "<li>🛠️ development/ - Active development work</li>",
            "<li>🧪 testing/ - Comprehensive testing framework</li>",
            "<li>🐛 debugging/ - Debug support and logging</li>",
            "<li>🔬 experiments/ - Experiment management</li>",
            "<li>✅ verification/ - Verification and validation</li>",
            "<li>📊 data/ - Data management</li>",
            "<li>📈 results/ - Results and outputs</li>",
            "<li>📚 documentation/ - Comprehensive documentation</li>",
            "</ul>",
            "</body></html>"
        ])
        
        with open(report_path, 'w') as f:
            f.write("\\n".join(html_content))
        
        self.log(f"📄 Setup report saved to: {report_path}")
        return str(report_path)
    
    def run_complete_setup(self) -> bool:
        """Run complete project setup."""
        self.log("🚀 Starting FinRL Contest 2024 Project Setup...")
        self.log("=" * 60)
        
        success = True
        
        # Check prerequisites
        if not self.check_prerequisites():
            success = False
        
        # Setup environments
        if success and not self.setup_environments():
            success = False
        
        # Create config templates
        if success and not self.create_config_templates():
            success = False
        
        # Create example scripts
        if success and not self.create_example_scripts():
            success = False
        
        # Run validation tests
        if success and not self.run_validation_tests():
            success = False
        
        self.log("=" * 60)
        
        if success:
            self.log("✅ Project setup completed successfully!")
        else:
            self.log("❌ Project setup completed with errors", "ERROR")
        
        # Generate report
        report_path = self.generate_setup_report()
        
        return success

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="FinRL Contest 2024 Project Setup")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--skip-env", action="store_true", help="Skip virtual environment setup")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation tests")
    
    args = parser.parse_args()
    
    setup = ProjectSetup(args.project_root)
    
    success = setup.run_complete_setup()
    
    if success:
        print("\\n🎉 Setup completed! Check the setup report for next steps.")
        sys.exit(0)
    else:
        print("\\n❌ Setup failed. Check the setup report for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()