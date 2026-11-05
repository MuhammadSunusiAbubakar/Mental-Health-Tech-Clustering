"""
DLBDSMLUSL01 - Task 1: Mental Health in Tech
Phase 1: Data Exploration
Author: Muhammad Sunusi Abubakar
"""
import pandas as pd

file_name = 'mental-health-in-tech-2016.csv'

try:
    df = pd.read_csv(file_name)
    print("--- 1. DATASET DIMENSIONS ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
    
    print("--- 2. DATA TYPES ---")
    print(df.dtypes.value_counts(), "\n")
    
    print("--- 3. MISSING VALUES ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
    
    missing_features = missing_df[missing_df['Count'] > 0].sort_values(by='Percentage', ascending=False)
    
    if not missing_features.empty:
        print(missing_features.head(15))
    else:
        print("No missing values.")
        
except FileNotFoundError:
    print(f"Error: '{file_name}' not found. Please place it in the current directory.")