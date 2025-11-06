"""
Convert cogging data files to Combined_book4.csv format

This script takes files like 085_2U_1132001_original_rotor_and_stator 1.csv or Syntec Motor.csv
and converts them to the format required by integrated_FFT_GUI.py (similar to Combined_book4.csv).

The conversion process matches cogging.py:
1. Reads X, CH1 (Torque/Signal), CH2 (Sine), CH3 (Cosine) from input file
2. Removes DC offset from CH2 and CH3
3. Computes angle using arctan2(CH2, CH3)
4. Unwraps the angle for continuous rotation
5. Converts to mechanical angle (degrees) - assumes 5000 sine/cosine cycles per revolution
6. Outputs: position4 (accumulated angle), CH4 (signal from CH1), X (sequence)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt



def tan_angle_to_mechanical_angle(cycles_per_rev, input_file, output_file):
    
    print(f"Reading input file: {input_file}")
    
    # Read CSV while skipping the second row and selecting CH1, CH2, CH3 (same as cogging.py)
    df = pd.read_csv(input_file, skiprows=[1], usecols=[0, 1, 2, 3])  # X, CH1 (Torque), CH2 (Sine), CH3 (Cosine)
    
    # Convert columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Remove DC offset from CH2 and CH3
    df["CH2"] -= df["CH2"].mean()
    df["CH3"] -= df["CH3"].mean()
    
    # Compute raw angles using arctan2
    df["Theta (radians)"] = np.arctan2(df["CH2"], df["CH3"])
    
    # Unwrap phase to ensure continuous mechanical rotation
    df["Unwrapped Theta (radians)"] = np.unwrap(df["Theta (radians)"])
    
    # Convert to mechanical angle (5000 sine/cosine cycles per revolution)
    df["Accumulated Theta (degrees)"] = (df["Unwrapped Theta (radians)"] / (2 * np.pi)) * (360 / cycles_per_rev)
    
    print(f"Processed {len(df)} rows")
    print(f"Angle range: {df['Accumulated Theta (degrees)'].min():.6f} to {df['Accumulated Theta (degrees)'].max():.6f} degrees")
    
    # Create output DataFrame in Combined_book4.csv format
    output_df = pd.DataFrame({
        'position4': df["Accumulated Theta (degrees)"],
        'CH4': df.iloc[:, 1],  # CH1 (the signal/torque)
        'X': df.iloc[:, 0]  # X (sequence)
    })
    
    # Write output file with metadata row (similar to Combined_book4.csv format)
    print(f"\nWriting output file: {output_file}")
    
    with open(output_file, 'w', newline='') as f:
        # Header row
        f.write("position4,CH4,X\n")
        # Metadata row (units)
        f.write("degree,Volt,Sequence\n")
        # Data rows
        output_df.to_csv(f, index=False, header=False, lineterminator='\n')
    
    print(f"✓ Successfully created: {output_file}")
    print(f"  Output shape: {output_df.shape}")
    
    return df, output_df # Exports dataframe for plotting functions


def data_plotter(df, output_df, show_xy_plot, show_polar_plot):
    """
    If plotter variables are enabled, then will plot and display the data.
    """
    # Plotting
    position4 = output_df['position4'].values
    CH4 = output_df['CH4'].values
    
    # X-Y Plot: Signal vs Accumulated Theta
    if show_xy_plot:
        plt.figure(figsize=(12, 5))
        plt.plot(position4, CH4, label=f"{Path(input_file).stem}: CH4 vs. Accumulated Theta", linestyle="-")
        plt.xlabel("Accumulated Theta (Degrees)")
        plt.ylabel("Signal (CH4)")
        plt.title("Signal vs. Accumulated Theta")
        plt.legend()
        plt.grid()
        plt.show()
    
    # Polar Plot: Signal vs Mechanical Angle
    if show_polar_plot:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
        ax.plot(np.radians(position4), CH4, label=f"{Path(input_file).stem}: CH4")
        ax.set_title("Polar Plot of Signal vs. Mechanical Angle")
        ax.legend()
        plt.show()
    
    else:
        return None


if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION - Edit these variables
    # ============================================================================
    input_file = "085_2U_1132001_original_rotor_and_stator 1.csv"
    output_file = "085_2U_1132001_original_combined.csv"
    cycles_per_rev = 5000  # Number of sine/cosine cycles per mechanical revolution
    
    # Plotting options
    show_xy_plot = True   # Set to True to show X-Y plot (Signal vs Accumulated Theta)
    show_polar_plot = True  # Set to True to show polar/circular plot
    # ============================================================================
    
    # Perform conversion
    df, output_df = tan_angle_to_mechanical_angle(cycles_per_rev, input_file, output_file)
    
    data_plotter(df, output_df, show_xy_plot, show_polar_plot)