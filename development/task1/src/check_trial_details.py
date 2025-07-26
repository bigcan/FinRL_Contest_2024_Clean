#!/usr/bin/env python3
"""
Check detailed trial information from HPO database
"""
import sqlite3
import json

def check_trial_details():
    conn = sqlite3.connect("task1_production_hpo.db")
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Check trial_user_attributes for stored metrics
    print("\nChecking trial_user_attributes table...")
    cursor.execute("SELECT trial_id, key, value_json FROM trial_user_attributes LIMIT 20")
    attrs = cursor.fetchall()
    
    if attrs:
        print(f"Found {len(attrs)} user attributes:")
        for trial_id, key, value in attrs:
            try:
                parsed_value = json.loads(value)
                print(f"  Trial {trial_id}: {key} = {parsed_value}")
            except:
                print(f"  Trial {trial_id}: {key} = {value}")
    
    # Check trial_intermediate_values for training progress
    print("\nChecking trial_intermediate_values table...")
    cursor.execute("SELECT trial_id, step, intermediate_value FROM trial_intermediate_values ORDER BY intermediate_value DESC LIMIT 10")
    intermediates = cursor.fetchall()
    
    if intermediates:
        print(f"Top intermediate values:")
        for trial_id, step, value in intermediates:
            print(f"  Trial {trial_id}, Step {step}: {value}")
    
    # Check actual trial values
    print("\nChecking trial_values table...")
    cursor.execute("SELECT trial_id, value FROM trial_values WHERE value != '-inf' LIMIT 10")
    values = cursor.fetchall()
    
    if values:
        print(f"Non-inf trial values found:")
        for trial_id, value in values:
            print(f"  Trial {trial_id}: {value}")
    else:
        print("  No non-inf values found in trial_values table")
    
    conn.close()

if __name__ == "__main__":
    check_trial_details()