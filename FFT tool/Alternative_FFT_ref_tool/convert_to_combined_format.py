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


"""
2 different approaches - one using the zero crossing method, and one using the arctan method.
"""


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


def data_plotter(df, output_df, input_file, show_xy_plot, show_polar_plot):
    """
    If plotter variables are enabled, then will plot and display the data.
    
    Args:
        df: Original dataframe
        output_df: Output dataframe with position4, CH4, X
        input_file: Input filename (for labeling plots)
        show_xy_plot: If True, show X-Y plot
        show_polar_plot: If True, show polar plot
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
    
    return None


def zero_crossing_angle_conversion(input_file, output_file, show_plots=False, x_min=None, x_max=None):
    """
    Convert data using zero-crossing detection method.
    
    This function:
    1. Reads CH2 (sine) and CH3 (cosine) data
    2. Removes DC offset (mean value) from both
    3. Detects zero crossings (sign changes)
    4. Creates progressive angles based on zero crossings
    5. Assumes constant speed between zero crossings
    6. d_theta per zero crossing = 0.036 degrees (360/10000)
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        show_plots: If True, plots sine and cosine with zero crossing markers
        x_min: Optional minimum X value to truncate data (None = no truncation)
        x_max: Optional maximum X value to truncate data (None = no truncation)
    
    Returns:
        output_df: DataFrame with position4 (angles), CH4 (signal), X (sequence)
    """
    print(f"\n{'='*70}")
    print("Zero-Crossing Angle Conversion")
    print(f"{'='*70}")
    print(f"Reading input file: {input_file}")
    
    # Read CSV while skipping the second row and selecting CH1, CH2, CH3
    df = pd.read_csv(input_file, skiprows=[1], usecols=[0, 1, 2, 3])  # X, CH1, CH2, CH3
    
    # Convert columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Extract columns
    X = df.iloc[:, 0].values  # Sequence
    CH1 = df.iloc[:, 1].values  # Signal/Torque
    CH2 = df.iloc[:, 2].values  # Sine
    CH3 = df.iloc[:, 3].values  # Cosine
    
    # Truncate data based on X limits (if specified) to speed up processing/plotting
    if x_min is not None or x_max is not None:
        mask = np.ones(len(X), dtype=bool)
        if x_min is not None:
            mask = mask & (X >= x_min)
            print(f"Truncating: X >= {x_min}")
        if x_max is not None:
            mask = mask & (X <= x_max)
            print(f"Truncating: X <= {x_max}")
        
        X = X[mask]
        CH1 = CH1[mask]
        CH2 = CH2[mask]
        CH3 = CH3[mask]
        print(f"Data truncated: {len(X)} points remaining (from {len(df)} original points)")
    
    # Find and print mean values (DC offsets)
    CH2_mean = np.mean(CH2)
    CH3_mean = np.mean(CH3)
    print(f"\nDC Offsets (mean values):")
    print(f"  CH2 (sine) mean: {CH2_mean:.6f}")
    print(f"  CH3 (cosine) mean: {CH3_mean:.6f}")
    
    # Remove DC offset (subtractive offset)
    CH2_centered = CH2 - CH2_mean
    CH3_centered = CH3 - CH3_mean
    
    # Detect zero crossings for both sine and cosine
    # Zero crossing occurs when signal crosses zero - need to interpolate exact positions
    def find_zero_crossings_interpolated(x_values, signal):

        """

        Possible zero crossing function to use: https://librosa.org/doc/0.11.0/generated/librosa.zero_crossings.html
        CHECK HERE


        Find exact zero crossing positions by interpolating between points where sign changes.
        Returns the interpolated X positions where signal = 0.
        """
        zero_crossing_x = []
        zero_crossing_indices = []
        
        # Find where sign changes (zero crossing occurs between these points)
        for i in range(len(signal) - 1):
            if np.sign(signal[i]) != np.sign(signal[i + 1]):
                # Linear interpolation to find exact zero crossing
                # x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
                x1, y1 = x_values[i], signal[i]
                x2, y2 = x_values[i + 1], signal[i + 1]
                
                # Avoid division by zero (shouldn't happen if signs are different, but be safe)
                if abs(y2 - y1) > 1e-10:
                    x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
                    zero_crossing_x.append(x_zero)
                    # Store the index of the point just before the zero crossing for reference
                    zero_crossing_indices.append(i)
        
        return np.array(zero_crossing_x), np.array(zero_crossing_indices)
    
    # Find interpolated zero crossings for sine (CH2) and cosine (CH3)
    zc_x_sine, zc_idx_sine = find_zero_crossings_interpolated(X, CH2_centered)
    zc_x_cosine, zc_idx_cosine = find_zero_crossings_interpolated(X, CH3_centered)
    
    print(f"\nZero crossings detected (interpolated to exact y=0 positions):")
    print(f"  CH2 (sine) zero crossings: {len(zc_x_sine)}")
    print(f"  CH3 (cosine) zero crossings: {len(zc_x_cosine)}")
    
    """

    ANCHOR| Location where the zero crossing list is created and sorted.
    
    We can either: Combine zero crossings from both signals and sort by X position, OR 
    select from ONE of the signals only. This is best, to avoid double counting zero crossings.
    
    all_zero_crossings_x = np.concatenate([[X[0]], zc_x_sine, [X[-1]]]) -> For this, only the sine signal is used.
    """
    # Combine zero crossings from both signals and sort by X position
    # all_zero_crossings_x = np.unique(np.concatenate([zc_x_sine, zc_x_cosine]))
    # all_zero_crossings_x = np.sort(all_zero_crossings_x)

    all_zero_crossings_x = np.concatenate([[X[0]], zc_x_sine, [X[-1]]])
    all_zero_crossings_x = np.sort(np.unique(all_zero_crossings_x))
    
    # Ensure we start from beginning and end at the end
    if all_zero_crossings_x[0] > X[0]:
        all_zero_crossings_x = np.concatenate([[X[0]], all_zero_crossings_x])
    if all_zero_crossings_x[-1] < X[-1]:
        all_zero_crossings_x = np.concatenate([all_zero_crossings_x, [X[-1]]])
    
    print(f"  Total unique zero crossings (including start/end): {len(all_zero_crossings_x)}")
    
    # Convert zero crossing X positions to indices for angle assignment
    # We need to find which data point indices correspond to these interpolated X positions
    # For angle assignment, we'll use the nearest data point indices
    all_zero_crossing_indices = []
    for x_zc in all_zero_crossings_x:
        # Find the index of the data point closest to this zero crossing X position
        idx = np.argmin(np.abs(X - x_zc))
        all_zero_crossing_indices.append(idx)
    all_zero_crossing_indices = np.unique(np.array(all_zero_crossing_indices))
    all_zero_crossing_indices = np.sort(all_zero_crossing_indices)
    
    # Plotting: Show sine and cosine with zero crossing markers at interpolated positions
    if show_plots:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot CH2 (sine) with interpolated zero crossings
        ax1.plot(X, CH2_centered, 'b-', label='CH2 (sine, DC removed)', linewidth=0.8)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        # Plot zero crossings at interpolated X positions (y=0)
        ax1.scatter(zc_x_sine, np.zeros_like(zc_x_sine), color='red', marker='o', 
                   s=50, zorder=5, label=f'Zero crossings ({len(zc_x_sine)})')
        ax1.set_xlabel('X (Sequence)')
        ax1.set_ylabel('CH2 (Sine, DC removed)')
        ax1.set_title(f'{Path(input_file).stem}: Sine Signal with Zero Crossings (interpolated)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot CH3 (cosine) with interpolated zero crossings
        ax2.plot(X, CH3_centered, 'g-', label='CH3 (cosine, DC removed)', linewidth=0.8)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        # Plot zero crossings at interpolated X positions (y=0)
        ax2.scatter(zc_x_cosine, np.zeros_like(zc_x_cosine), color='red', marker='o', 
                   s=50, zorder=5, label=f'Zero crossings ({len(zc_x_cosine)})')
        ax2.set_xlabel('X (Sequence)')
        ax2.set_ylabel('CH3 (Cosine, DC removed)')
        ax2.set_title(f'{Path(input_file).stem}: Cosine Signal with Zero Crossings (interpolated)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Create progressive angles based on zero crossings
    # d_theta per zero crossing = 0.036 degrees (360/10000)
    d_theta_per_zc = 0.036  # degrees
    
    print(f"\nAngle increment per zero crossing: {d_theta_per_zc} degrees")
    print(f"Creating progressive angles...")
    
    # Initialize angle array
    position4 = np.zeros(len(X))
    
    # Process each interval between zero crossing indices
    # We use the indices to assign angles to data points
    for i in range(len(all_zero_crossing_indices) - 1):
        start_idx = all_zero_crossing_indices[i]
        end_idx = all_zero_crossing_indices[i + 1]
        
        # Number of points in this interval
        n_points = end_idx - start_idx + 1
        
        # Cumulative angle at start of this interval (i-th zero crossing)
        cumulative_angle_start = i * d_theta_per_zc
        
        # Cumulative angle at end of this interval ((i+1)-th zero crossing)
        cumulative_angle_end = (i + 1) * d_theta_per_zc
        
        # Interpolate angles linearly across this interval (constant speed assumption)
        # This distributes angles evenly between zero crossings
        if n_points > 1:
            angles_in_interval = np.linspace(cumulative_angle_start, cumulative_angle_end, n_points)
        else:
            angles_in_interval = np.array([cumulative_angle_start])
        
        # Assign angles to this interval
        position4[start_idx:end_idx+1] = angles_in_interval
    
    print(f"Angle range: {position4[0]:.6f} to {position4[-1]:.6f} degrees")
    print(f"Total rotation: {position4[-1] - position4[0]:.6f} degrees")
    
    # Create output DataFrame in Combined_book4.csv format
    output_df = pd.DataFrame({
        'position4': position4,
        'CH4': CH1,  # CH1 (the signal/torque)
        'X': X  # X (sequence)
    })
    
    # Write output file with metadata row
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
    
    return output_df


if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION - Edit these variables
    # ============================================================================
    
    # Method 1: Arctan2-based conversion (original method)
    use_arctan_method = True
    input_file_arctan = "Syntec Motor.csv"#"085_2U_1132001_original_rotor_and_stator 1.csv"
    output_file_arctan = "Syntec Motor_arctan_method.csv" #"085_2U_1132001_original_arctan_method.csv"
    cycles_per_rev = 10000  # Number of sine/cosine cycles per mechanical revolutio - this will depend on the type of encoder used
    
    # Plotting options for arctan method
    show_xy_plot = True   # Set to True to show X-Y plot (Signal vs Accumulated Theta)
    show_polar_plot = True  # Set to True to show polar/circular plot
    
    # Method 2: Zero-crossing based conversion
    use_zero_crossing_method = True
    input_file_zc = "Syntec Motor.csv"#"085_2U_1132001_original_rotor_and_stator 1.csv"
    output_file_zc = "Syntec Motor_zero_cross_sine_only.csv" #"085_2U_1132001_zero_cross_sine_only.csv"
    show_zc_plots = False # Set to True to show sine/cosine plots with zero crossing markers - if producing the full plot, set thi to false due to plot rendering beiing very intensive due to many datapoints.
    
    # X-value truncation (to speed up plotting/processing)
    # Set to None to disable truncation, or specify numeric values
    x_min_truncate = 0 # e.g., 0 or None for no minimum limit
    x_max_truncate = None  # e.g., 10000 or None for no maximum limit
    # ============================================================================
    
    # Perform arctan2-based conversion
    if use_arctan_method:
        df, output_df = tan_angle_to_mechanical_angle(cycles_per_rev, input_file_arctan, output_file_arctan)
        data_plotter(df, output_df, input_file_arctan, show_xy_plot, show_polar_plot)
    
    # Perform zero-crossing based conversion
    if use_zero_crossing_method:
        zero_crossing_angle_conversion(input_file_zc, output_file_zc, show_plots=show_zc_plots, 
                                       x_min=x_min_truncate, x_max=x_max_truncate)