"""
TEMPORARY UTILITY FUNCTION
Adds time column to Combined_book4.csv
This function can be deleted after running once.

Useful for data parsed form the Oscilloscope USB file - due to being stored in mixed data format (annoying presense of Strings).
"""
import pandas as pd
import numpy as np

def add_time_column_to_csv():
    """Add time column to Combined_book4.csv and save it."""
    filename = 'Combined_book4.csv'
    
    print(f"Reading {filename}...")
    # Read CSV with first row as headers, skip the second row (metadata)
    df = pd.read_csv(filename, header=0, skiprows=[1])
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Data shape: {df.shape}")
    
    # Check if time column already exists
    if 'time' in df.columns:
        print("'time' column already exists. Skipping.")
        return
    
    # Add time column from X or Sequence
    if 'X' in df.columns:
        df['time'] = df['X'] * 0.00004
        print("Added 'time' column from X * 0.00004")
    elif 'Sequence' in df.columns:
        df['time'] = df['Sequence'] * 0.00004
        print("Added 'time' column from Sequence * 0.00004")
    else:
        print("ERROR: Neither 'X' nor 'Sequence' column found!")
        return
    
    # Read the original file to preserve the metadata row
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Reconstruct the file with metadata row
    print(f"\nWriting updated file...")
    
    # Write header
    header_line = lines[0].strip()
    metadata_line = lines[1].strip()
    
    # Add 'time' to header if not already there
    if 'time' not in header_line:
        header_line = header_line + ',time'
    
    # Add 'time' unit to metadata if not already there
    if 'time' not in metadata_line.lower():
        # Find what the last unit is and add appropriate unit
        metadata_parts = metadata_line.split(',')
        if len(metadata_parts) == len(header_line.split(',')) - 1:
            metadata_line = metadata_line + ',second'  # or appropriate unit
    
    # Write to a temporary file first, then rename (safer)
    temp_filename = filename + '.tmp'
    try:
        with open(temp_filename, 'w', newline='') as f:
            f.write(header_line + '\n')
            f.write(metadata_line + '\n')
            
            # Write data rows
            for idx, row in df.iterrows():
                row_values = [str(row[col]) for col in df.columns]
                f.write(','.join(row_values) + '\n')
        
        # Replace original file with temporary file
        import os
        import shutil
        shutil.move(temp_filename, filename)
    except Exception as e:
        print(f"Error writing file: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise
    
    print(f"Successfully updated {filename}")
    print(f"New columns: {df.columns.tolist()}")
    print(f"Time column range: {df['time'].min():.6f} to {df['time'].max():.6f}")
    print("\nThis function can now be deleted if desired.")

if __name__ == '__main__':
    add_time_column_to_csv()

