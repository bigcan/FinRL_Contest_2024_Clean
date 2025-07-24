"""
Update Configuration Files for Enhanced Features

Updates all relevant configuration files to use dynamic state_dim detection
from TradeSimulator instead of hardcoded values.
"""

import os
import re

def update_file(file_path):
    """Update a single file to use dynamic state_dim"""
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Pattern to match state_dim = 8 + 2 or similar
        pattern = r'([\s\.]state_dim\s*=\s*)8\s*\+\s*2'
        
        if re.search(pattern, content):
            # Check if we already import TradeSimulator
            if 'from trade_simulator import' not in content and 'import trade_simulator' not in content:
                # Add import at the top (after existing imports)
                import_pattern = r'(import\s+\w+.*\n|from\s+\w+.*\n)+'
                import_match = re.search(import_pattern, content)
                if import_match:
                    import_end = import_match.end()
                    content = (content[:import_end] + 
                             'from trade_simulator import TradeSimulator\n' + 
                             content[import_end:])
            
            # Replace hardcoded state_dim with dynamic detection
            replacement = r'\1self._get_dynamic_state_dim()'
            content = re.sub(pattern, replacement, content)
            
            # Add helper method to get state_dim
            if 'def _get_dynamic_state_dim(self):' not in content:
                # Find a good place to add the method (after __init__ or before train)
                class_pattern = r'(class\s+\w+.*?:.*?\n(?:\s{4}.*\n)*?)'
                init_pattern = r'(\s+def\s+__init__.*?(?:\n\s{8}.*)*\n)'
                
                helper_method = '''
    def _get_dynamic_state_dim(self):
        """Get state dimension from TradeSimulator"""
        try:
            temp_sim = TradeSimulator(num_sims=1)
            return temp_sim.state_dim
        except:
            return 10  # Fallback to original value
'''
                
                # Try to insert after __init__
                init_match = re.search(init_pattern, content)
                if init_match:
                    insert_pos = init_match.end()
                    content = content[:insert_pos] + helper_method + content[insert_pos:]
                else:
                    # Insert after class definition
                    class_match = re.search(class_pattern, content)
                    if class_match:
                        insert_pos = class_match.end()
                        content = content[:insert_pos] + helper_method + content[insert_pos:]
            
            # Write back the updated content
            with open(file_path, 'w') as f:
                f.write(content)
            
            return True
    
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False
    
    return False

def main():
    """Update all configuration files"""
    print("=" * 60)
    print("UPDATING CONFIGS FOR ENHANCED FEATURES")
    print("=" * 60)
    
    # Files to update (main ones used)
    files_to_update = [
        'task1_ensemble.py',
        'task1_eval.py',
        'task1_eval_reward_shaped.py',
        'task1_eval_exploratory.py',
        'debug_ensemble_eval.py'
    ]
    
    updated_count = 0
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            print(f"Updating {file_path}...")
            if update_file(file_path):
                print(f"✓ Updated {file_path}")
                updated_count += 1
            else:
                print(f"- No changes needed for {file_path}")
        else:
            print(f"✗ File not found: {file_path}")
    
    print("\n" + "=" * 60)
    print(f"CONFIGURATION UPDATE COMPLETE")
    print("=" * 60)
    print(f"✓ Updated {updated_count} files")
    print("✓ Configurations now use dynamic state_dim detection")
    print("✓ Will automatically adapt to enhanced vs original features")

if __name__ == "__main__":
    main()