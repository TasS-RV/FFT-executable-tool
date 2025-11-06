import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tkinter as tk
from tkinter import ttk


def read_csv_data(filename):
    """Read the CSV file and extract all data, adding time column if needed."""
    # Read CSV with first row as headers, skip the second row (metadata with units)
    # Read all columns first to get available column names
    df_full = pd.read_csv(filename, header=0, skiprows=[1])
    
    # Get available columns
    available_columns = df_full.columns.tolist()
    print(f"Available columns: {available_columns}")
    
    # Add time column if X or Sequence column exists AND time doesn't already exist
    if 'time' not in available_columns:
        if 'X' in available_columns:
            df_full['time'] = df_full['X'] * 0.00004
            print("Added 'time' column from X * 0.00004")
        elif 'Sequence' in available_columns:
            df_full['time'] = df_full['Sequence'] * 0.00004
            print("Added 'time' column from Sequence * 0.00004")
        
        # Update available columns after adding time
        available_columns = df_full.columns.tolist()
    else:
        print("'time' column already exists in CSV file")
    
    # For backward compatibility, try to get position and CH4 if they exist
    # Look for position column (exact match or contains 'position')
    position = None
    CH4 = None
    position_col = None
    
    # Try exact match first
    if 'position' in available_columns:
        position_col = 'position'
    else:
        # Look for columns containing 'position'
        for col in available_columns:
            if 'position' in col.lower():
                position_col = col
                break
    
    if position_col is not None:
        position = df_full[position_col].values
    if 'CH4' in available_columns:
        CH4 = df_full['CH4'].values
    
    # Calculate base increment from position differences if position exists
    base_increment = None
    if position is not None and len(position) > 1:
        position_diff = np.diff(position)
        unique_diffs = np.unique(position_diff)
        positive_diffs = unique_diffs[unique_diffs > 0]
        if len(positive_diffs) > 0:
            base_increment = np.min(positive_diffs)
    
    return df_full, available_columns, base_increment

def compute_fft(signal, dx, freq_start=0.0, freq_end=None):
    """Compute FFT of signal with given step size dx and optional frequency limits.
    
    Args:
        signal: Input signal array
        dx: Step size
        freq_start: Start frequency (default 0.0)
        freq_end: End frequency (None means use Nyquist/max frequency)
    """
    N = len(signal)
    
    # Compute FFT
    fft_vals = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_vals)
    
    # Frequency axis in terms of position (spatial frequency)
    frequencies = np.fft.fftfreq(N, dx)
    
    # Return only positive frequencies
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    fft_magnitude = fft_magnitude[positive_freq_idx]
    
    # Apply frequency start limit (default to 0)
    if freq_start is not None and freq_start > 0:
        freq_mask = frequencies >= freq_start
        frequencies = frequencies[freq_mask]
        fft_magnitude = fft_magnitude[freq_mask]
    
    # Apply frequency end limit if specified
    if freq_end is not None:
        freq_mask = frequencies <= freq_end
        frequencies = frequencies[freq_mask]
        fft_magnitude = fft_magnitude[freq_mask]
    
    return frequencies, fft_magnitude

def create_plots(df, x_col, y_col, dx, base_increment, freq_start, freq_end, y_min=None, y_max=None, manual_y_min=None, manual_y_max=None, manual_freq_start=None, manual_freq_end=None):
    """Create or recreate FFT and data plots."""
    # MANUAL Y-AXIS LIMIT VARIABLE - This is the hard limit that will be enforced
    # Use manual_y_min/manual_y_max if provided (overrides UI), otherwise use function parameters (from UI)
    if manual_y_max is None:
        manual_y_max = y_max
    
    if manual_y_min is None:
        manual_y_min = y_min
    
    # MANUAL FREQUENCY LIMIT VARIABLE - Override frequency limits if provided
    if manual_freq_start is not None:
        freq_start = manual_freq_start
    
    if manual_freq_end is not None:
        freq_end = manual_freq_end
    
    # Close any existing matplotlib windows
    plt.close('all')
    
    # Create new figure with only plots (no controls)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    ax_fft = fig.add_subplot(gs[0])
    ax_data = fig.add_subplot(gs[1])
    
    # Disable Y-axis autoscaling IMMEDIATELY if manual limits are provided
    # This must be done before any plotting to prevent matplotlib from auto-scaling
    if manual_y_min is not None or manual_y_max is not None:
        ax_fft.set_autoscaley_on(False)
        ax_fft.autoscale(enable=False, axis='y')
    
    # Get data from selected columns and filter out NaN values
    x_data = df[x_col].values
    y_data = df[y_col].values
    
    # Filter out NaN values - keep only valid data points
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    
    # Always calculate dx from the actual x_data being used
    # This ensures correct scaling regardless of which column is selected
    x_diff = np.diff(x_data)
    unique_diffs = np.unique(x_diff)
    positive_diffs = unique_diffs[unique_diffs > 0]
    
    if len(positive_diffs) > 0:
        # Use the most common (smallest) step size
        base_step = np.min(positive_diffs)
        # Apply the multiplier
        current_dx = base_step * (dx / base_increment if base_increment is not None and base_increment > 0 else 1.0)
    else:
        # Fallback: use provided dx or calculate from base_increment
        if base_increment is not None:
            current_dx = dx
        else:
            current_dx = 1.0
    
    print(f"\nCreating plots with:")
    print(f"  X column: {x_col}, dx: {current_dx:.6f}")
    print(f"  X data range: {np.min(x_data):.6f} to {np.max(x_data):.6f}")
    print(f"  Y data range: {np.min(y_data):.6f} to {np.max(y_data):.6f}")
    print(f"  Frequency range: {freq_start} to {freq_end if freq_end is not None else 'max'}")
    
    # Compute FFT
    freq, fft_mag = compute_fft(y_data, current_dx, freq_start, freq_end)
    
    print(f"\nFFT Results:")
    print(f"  Number of frequency points: {len(freq)}")
    print(f"  Frequency range: {freq[0]:.6f} to {freq[-1]:.6f} (if available)")
    print(f"  FFT magnitude range: {np.min(fft_mag):.6f} to {np.max(fft_mag):.6f}")
    if len(freq) > 0:
        print(f"  First 5 frequencies: {freq[:5]}")
        print(f"  Last 5 frequencies: {freq[-5:]}")
        print(f"  First 5 magnitudes: {fft_mag[:5]}")
        print(f"  Last 5 magnitudes: {fft_mag[-5:]}")
    print()
    
    # Plot FFT
    ax_fft.plot(freq, fft_mag, 'r-', linewidth=1.5)
    ax_fft.set_title('FFT Plot', fontsize=12, fontweight='bold')
    ax_fft.set_xlabel(f'Spatial Frequency (1/{x_col})', fontsize=10)
    ax_fft.set_ylabel('Magnitude', fontsize=10)
    ax_fft.grid(True, alpha=0.3)
    
    # IMMEDIATELY after plotting, if manual limits are set, apply them NOW
    # This prevents matplotlib from auto-scaling based on the plot data
    if manual_y_max is not None or manual_y_min is not None:
        ax_fft.set_autoscaley_on(False)
        ax_fft.autoscale(enable=False, axis='y')
        if manual_y_max is not None and manual_y_min is not None:
            ax_fft.set_ylim(manual_y_min, manual_y_max)
        elif manual_y_max is not None:
            # Get the current bottom from the plot, then set top to manual_y_max
            current_bottom = ax_fft.get_ylim()[0]
            ax_fft.set_ylim(current_bottom, manual_y_max)
        elif manual_y_min is not None:
            current_top = ax_fft.get_ylim()[1]
            ax_fft.set_ylim(manual_y_min, current_top)
    
    # Plot data
    ax_data.plot(x_data, y_data, 'b-', linewidth=0.5, alpha=0.7)
    ax_data.set_title('Data Plot', fontsize=12, fontweight='bold')
    ax_data.set_xlabel(f'{x_col}', fontsize=10)
    ax_data.set_ylabel(f'{y_col}', fontsize=10)
    ax_data.grid(True, alpha=0.3)
    
    # Set data axis limits with NaN/Inf checks
    # Note: Use different variable names to avoid conflict with FFT y_min/y_max parameters
    x_min_data, x_max_data = np.min(x_data), np.max(x_data)
    y_min_data, y_max_data = np.min(y_data), np.max(y_data)
    
    if np.isfinite(x_min_data) and np.isfinite(x_max_data):
        ax_data.set_xlim(x_min_data, x_max_data)
    else:
        ax_data.set_xlim(0, len(x_data))
    
    if np.isfinite(y_min_data) and np.isfinite(y_max_data):
        ax_data.set_ylim(y_min_data * 1.05, y_max_data * 1.05)
    else:
        ax_data.set_ylim(-1, 1)
    
    plt.suptitle('FFT Analysis Tool', fontsize=14, fontweight='bold')
    
    # NOW set all axis limits AFTER all plotting is complete
    # This ensures matplotlib doesn't override our manual limits
    
    # Set FFT X-axis limits
    if len(freq) > 0 and len(fft_mag) > 0:
        x_min = freq[0]
        x_max = freq[-1]
        
        if freq_start is not None:
            x_min = max(x_min, freq_start)
        if freq_end is not None:
            x_max = min(x_max, freq_end)
        
        ax_fft.set_xlim(x_min, x_max)
        
        # Get magnitude range for reference
        max_mag = np.max(fft_mag)
        min_mag = np.min(fft_mag)
        
        print(f"\nY-Axis Limit Settings:")
        print(f"  Manual y_min: {manual_y_min}, Manual y_max: {manual_y_max}")
        print(f"  FFT magnitude range: {min_mag:.6f} to {max_mag:.6f}")
        
        # CRITICAL: Ensure autoscaling is OFF when manual limits are provided
        if manual_y_min is not None or manual_y_max is not None:
            ax_fft.set_autoscaley_on(False)
            ax_fft.autoscale(enable=False, axis='y')
        
        # Set Y-axis limits - using manual limits
        if manual_y_min is not None and manual_y_max is not None:
            # Both limits provided - use them exactly
            print(f"  Setting Y-axis limits to: [{manual_y_min}, {manual_y_max}]")
            ax_fft.set_ylim(manual_y_min, manual_y_max)
            ax_fft.set_autoscaley_on(False)
        elif manual_y_min is not None:
            # Only y_min provided, use auto for max
            if np.isfinite(max_mag) and max_mag > 0:
                y_max_auto = max_mag * 1.1
                print(f"  Setting Y-axis limits to: [{manual_y_min}, {y_max_auto:.6f}] (auto max)")
            else:
                y_max_auto = 1.0
                print(f"  Setting Y-axis limits to: [{manual_y_min}, 1.0] (fallback)")
            ax_fft.set_ylim(manual_y_min, y_max_auto)
            ax_fft.set_autoscaley_on(False)
        elif manual_y_max is not None:
            # Only y_max provided - TRUNCATE AT y_max - THIS IS THE KEY CASE
            # Get current min from data to preserve the lower range
            current_y_min = 0.0 if min_mag >= 0 else min_mag * 1.1
            print(f"  TRUNCATING Y-axis at y_max={manual_y_max}: Setting Y-axis limits to: [{current_y_min:.6f}, {manual_y_max}]")
            ax_fft.set_ylim(current_y_min, manual_y_max)
            # Force autoscale OFF
            ax_fft.set_autoscaley_on(False)
            ax_fft.autoscale(enable=False, axis='y')
        else:
            # Auto-scale Y-axis (no manual limits)
            ax_fft.set_autoscaley_on(True)
            if np.isfinite(max_mag) and max_mag > 0:
                y_max_auto = max_mag * 1.1
                print(f"  Auto-scaling Y-axis to: [0, {y_max_auto:.6f}]")
                ax_fft.set_ylim(0, y_max_auto)
            else:
                print(f"  Setting Y-axis limits to: [0, 1.0] (fallback)")
                ax_fft.set_ylim(0, 1.0)
    else:
        ax_fft.set_xlim(0, 1)
        ax_fft.set_ylim(0, 1)
    
    # FINAL: Re-apply Y-axis limits one more time to ensure they stick
    # This is necessary because sometimes matplotlib resets limits during show()
    if len(freq) > 0 and len(fft_mag) > 0:
        if manual_y_min is not None and manual_y_max is not None:
            # Both set, force both
            ax_fft.set_ylim(manual_y_min, manual_y_max)
            ax_fft.set_autoscaley_on(False)
            ax_fft.autoscale(enable=False, axis='y')
            print(f"  FINAL: Re-enforcing Y-axis limits: [{manual_y_min}, {manual_y_max}]")
        elif manual_y_max is not None:
            # If y_max is set, force it again (truncate at y_max)
            current_y_min = ax_fft.get_ylim()[0]
            ax_fft.set_ylim(current_y_min, manual_y_max)
            ax_fft.set_autoscaley_on(False)
            ax_fft.autoscale(enable=False, axis='y')
            print(f"  FINAL: Re-enforcing Y-axis limit at y_max={manual_y_max}")
        elif manual_y_min is not None:
            # If y_min is set, force it again
            current_y_max = ax_fft.get_ylim()[1]
            ax_fft.set_ylim(manual_y_min, current_y_max)
            ax_fft.set_autoscaley_on(False)
            ax_fft.autoscale(enable=False, axis='y')
            print(f"  FINAL: Re-enforcing Y-axis limit at y_min={manual_y_min}")
    
    # Verify the limits were set correctly
    ylim_check = ax_fft.get_ylim()
    print(f"  Final verified Y-axis limits: [{ylim_check[0]:.6f}, {ylim_check[1]:.6f}]")
    print(f"  Autoscaling enabled: {ax_fft.get_autoscaley_on()}")
    
    # Force the figure to update
    plt.figure(fig.number)
    plt.draw()
    
    plt.show(block=False)

def create_control_window(df, available_columns, default_x_col, default_y_col, 
                         base_increment, initial_dx, callback):
    """Create a tkinter popup window with controls."""
    root = tk.Tk()
    root.title("FFT Control Panel")
    root.geometry("400x600")
    
    # Variables
    x_col_var = tk.StringVar(value=default_x_col)
    y_col_var = tk.StringVar(value=default_y_col)
    freq_start_var = tk.StringVar(value="")
    freq_end_var = tk.StringVar(value="")
    y_min_var = tk.StringVar(value="")
    y_max_var = tk.StringVar(value="")
    slider_var = tk.IntVar(value=1)
    
    # Title
    title_label = tk.Label(root, text="FFT Analysis Controls", 
                          font=("Arial", 14, "bold"))
    title_label.pack(pady=10)
    
    # X Column selection
    x_frame = tk.Frame(root)
    x_frame.pack(pady=5, padx=10, fill=tk.X)
    tk.Label(x_frame, text="X Column:", font=("Arial", 10)).pack(side=tk.LEFT)
    x_combo = ttk.Combobox(x_frame, textvariable=x_col_var, values=available_columns, 
                          state="readonly", width=20)
    x_combo.pack(side=tk.LEFT, padx=5)
    
    # Y Column selection
    y_frame = tk.Frame(root)
    y_frame.pack(pady=5, padx=10, fill=tk.X)
    tk.Label(y_frame, text="Y Column:", font=("Arial", 10)).pack(side=tk.LEFT)
    y_combo = ttk.Combobox(y_frame, textvariable=y_col_var, values=available_columns, 
                          state="readonly", width=20)
    y_combo.pack(side=tk.LEFT, padx=5)
    
    # Frequency range
    freq_frame = tk.LabelFrame(root, text="Frequency Range", font=("Arial", 10))
    freq_frame.pack(pady=10, padx=10, fill=tk.X)
    
    freq_start_frame = tk.Frame(freq_frame)
    freq_start_frame.pack(pady=5, padx=5, fill=tk.X)
    tk.Label(freq_start_frame, text="Start:", width=8).pack(side=tk.LEFT)
    freq_start_entry = tk.Entry(freq_start_frame, textvariable=freq_start_var, width=15)
    freq_start_entry.pack(side=tk.LEFT, padx=5)
    tk.Label(freq_start_frame, text="(leave empty for 0)", font=("Arial", 8), 
            fg="gray").pack(side=tk.LEFT)
    
    freq_end_frame = tk.Frame(freq_frame)
    freq_end_frame.pack(pady=5, padx=5, fill=tk.X)
    tk.Label(freq_end_frame, text="End:", width=8).pack(side=tk.LEFT)
    freq_end_entry = tk.Entry(freq_end_frame, textvariable=freq_end_var, width=15)
    freq_end_entry.pack(side=tk.LEFT, padx=5)
    tk.Label(freq_end_frame, text="(leave empty for max)", font=("Arial", 8), 
            fg="gray").pack(side=tk.LEFT)
    
    # Y-axis limits for FFT plot
    y_axis_frame = tk.LabelFrame(root, text="FFT Y-Axis Limits", font=("Arial", 10))
    y_axis_frame.pack(pady=10, padx=10, fill=tk.X)
    
    y_min_frame = tk.Frame(y_axis_frame)
    y_min_frame.pack(pady=5, padx=5, fill=tk.X)
    tk.Label(y_min_frame, text="Y Min:", width=8).pack(side=tk.LEFT)
    y_min_entry = tk.Entry(y_min_frame, textvariable=y_min_var, width=15)
    y_min_entry.pack(side=tk.LEFT, padx=5)
    tk.Label(y_min_frame, text="(leave empty for auto)", font=("Arial", 8), 
            fg="gray").pack(side=tk.LEFT)
    
    y_max_frame = tk.Frame(y_axis_frame)
    y_max_frame.pack(pady=5, padx=5, fill=tk.X)
    tk.Label(y_max_frame, text="Y Max:", width=8).pack(side=tk.LEFT)
    y_max_entry = tk.Entry(y_max_frame, textvariable=y_max_var, width=15)
    y_max_entry.pack(side=tk.LEFT, padx=5)
    tk.Label(y_max_frame, text="(leave empty for auto)", font=("Arial", 8), 
            fg="gray").pack(side=tk.LEFT)
    
    # Slider for dx
    slider_frame = tk.LabelFrame(root, text="dx Step Size (multiple of base)", 
                                 font=("Arial", 10))
    slider_frame.pack(pady=10, padx=10, fill=tk.X)
    
    max_multiple = int(len(df) / 10) if len(df) > 10 else 100
    if base_increment is None:
        max_multiple = 100
    
    slider = tk.Scale(slider_frame, from_=1, to=max_multiple, orient=tk.HORIZONTAL,
                     variable=slider_var, length=350)
    slider.pack(pady=10, padx=10)
    
    slider_value_label = tk.Label(slider_frame, text="", font=("Arial", 9))
    slider_value_label.pack()
    
    def update_slider_label(*args):
        multiplier = slider_var.get()
        if base_increment is not None:
            new_dx = base_increment * multiplier
            slider_value_label.config(
                text=f"{multiplier} × {base_increment:.6f} = {new_dx:.6f}")
        else:
            new_dx = initial_dx * multiplier
            slider_value_label.config(
                text=f"{multiplier} × {initial_dx:.6f} = {new_dx:.6f}")
    
    slider_var.trace('w', update_slider_label)
    update_slider_label()
    
    # Run button
    def run_fft():
        print("\n" + "="*60)
        print("RUN FFT PRESSED - Processing Parameters")
        print("="*60)
        
        # Get values
        x_col = x_col_var.get()
        y_col = y_col_var.get()
        multiplier = slider_var.get()
        new_dx = (base_increment * multiplier) if base_increment is not None else (initial_dx * multiplier)
        
        print(f"Selected Columns:")
        print(f"  X Column: {x_col}")
        print(f"  Y Column: {y_col}")
        
        # Get actual data values and filter NaN
        x_data = df[x_col].values
        y_data = df[y_col].values
        
        # Filter out NaN values for statistics
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data_valid = x_data[valid_mask]
        y_data_valid = y_data[valid_mask]
        
        print(f"\nData Statistics:")
        if len(x_data_valid) > 0:
            print(f"  X Data - Min: {np.min(x_data_valid):.6f}, Max: {np.max(x_data_valid):.6f}, Mean: {np.mean(x_data_valid):.6f}")
            print(f"  X Data - First 5 values: {x_data_valid[:5]}")
            print(f"  X Data - Last 5 values: {x_data_valid[-5:]}")
        else:
            print(f"  X Data - No valid data points (all NaN)")
        if len(y_data_valid) > 0:
            print(f"  Y Data - Min: {np.min(y_data_valid):.6f}, Max: {np.max(y_data_valid):.6f}, Mean: {np.mean(y_data_valid):.6f}")
            print(f"  Y Data - First 5 values: {y_data_valid[:5]}")
            print(f"  Y Data - Last 5 values: {y_data_valid[-5:]}")
        else:
            print(f"  Y Data - No valid data points (all NaN)")
        
        # Calculate step size from X data (use valid data)
        x_diff = np.diff(x_data_valid) if len(x_data_valid) > 0 else np.diff(x_data)
        unique_diffs = np.unique(x_diff)
        positive_diffs = unique_diffs[unique_diffs > 0]
        
        if len(positive_diffs) > 0:
            calculated_base_step = np.min(positive_diffs)
            print(f"\nStep Size Calculation:")
            print(f"  Base step from X data: {calculated_base_step:.6f}")
            base_inc_str = f"{base_increment:.6f}" if base_increment is not None else "None"
            print(f"  Base increment (from position): {base_inc_str}")
            print(f"  Multiplier: {multiplier}")
            print(f"  Calculated dx: {new_dx:.6f}")
        else:
            print(f"\nStep Size: Using provided dx = {new_dx:.6f}")
        
        # Parse frequency inputs
        freq_start_val = None
        freq_end_val = None
        
        freq_start_str = freq_start_var.get().strip()
        if freq_start_str:
            try:
                freq_start_val = float(freq_start_str)
            except ValueError:
                print(f"  WARNING: Invalid start frequency: {freq_start_str}")
        
        freq_end_str = freq_end_var.get().strip()
        if freq_end_str:
            try:
                freq_end_val = float(freq_end_str)
            except ValueError:
                print(f"  WARNING: Invalid end frequency: {freq_end_str}")
        
        freq_start_to_use = freq_start_val if freq_start_val is not None else 0.0
        freq_end_to_use = freq_end_val
        
        print(f"\nFrequency Range:")
        print(f"  Start frequency: {freq_start_to_use if freq_start_to_use is not None else '0.0 (default)'}")
        print(f"  End frequency: {freq_end_to_use if freq_end_to_use is not None else 'max (default)'}")
        
        # Parse Y-axis limit inputs
        y_min_val = None
        y_max_val = None
        
        y_min_str = y_min_var.get().strip()
        if y_min_str:
            try:
                y_min_val = float(y_min_str)
            except ValueError:
                print(f"  WARNING: Invalid Y min: {y_min_str}")
        
        y_max_str = y_max_var.get().strip()
        if y_max_str:
            try:
                y_max_val = float(y_max_str)
            except ValueError:
                print(f"  WARNING: Invalid Y max: {y_max_str}")
        
        print(f"\nY-Axis Limits:")
        print(f"  Y Min: {y_min_val if y_min_val is not None else 'auto'}")
        print(f"  Y Max: {y_max_val if y_max_val is not None else 'auto'}")
        
        print("="*60 + "\n")
        
        # Call the callback with parameters
        callback(x_col, y_col, new_dx, freq_start_to_use, freq_end_to_use, y_min_val, y_max_val)
    
    run_button = tk.Button(root, text="Run FFT", command=run_fft, 
                          bg="lightblue", font=("Arial", 12, "bold"),
                          width=20, height=2)
    run_button.pack(pady=20)
    
    # Close button
    close_button = tk.Button(root, text="Close", command=root.quit,
                             font=("Arial", 10), width=15)
    close_button.pack(pady=5)
    
    return root

def main(csv_filename=None, manual_y_min=None, manual_y_max=None, manual_freq_start=None, manual_freq_end=None):
    # Read data
    global filename
    if csv_filename is not None:
        filename = csv_filename
    print(f"Reading data from {filename}...")
    df, available_columns, base_increment = read_csv_data(filename)
    
    print(f"Data loaded: {len(df)} points")
    if base_increment is not None:
        print(f"Base increment: {base_increment:.6f}")
    
    # Default column selections
    default_x_col = available_columns[0]
    for col in available_columns:
        if 'position' in col.lower():
            default_x_col = col
            break
    
    default_y_col = 'CH4' if 'CH4' in available_columns else available_columns[-1]
    
    # Initial values
    initial_dx = base_increment if base_increment is not None else 1.0
    
    # Determine initial frequency range (use manual if provided, otherwise defaults)
    initial_freq_start = manual_freq_start if manual_freq_start is not None else 0.0
    initial_freq_end = manual_freq_end
    
    # Create initial plots
    create_plots(df, default_x_col, default_y_col, initial_dx, base_increment, initial_freq_start, initial_freq_end, None, None, manual_y_min, manual_y_max, manual_freq_start, manual_freq_end)
    
    # Create control window
    control_window = create_control_window(
        df, available_columns, default_x_col, default_y_col,
        base_increment, initial_dx,
        lambda x_col, y_col, dx, freq_start, freq_end, y_min, y_max: create_plots(
            df, x_col, y_col, dx, base_increment, freq_start, freq_end, y_min, y_max, manual_y_min, manual_y_max, manual_freq_start, manual_freq_end)
    )
    
    # Run tkinter main loop
    control_window.mainloop()
    
    # Close matplotlib when control window closes
    plt.close('all')

def update_csv_with_time(filename):
    """
    TEMPORARY UTILITY FUNCTION
    Creates a new CSV file with time column calculated from X or Sequence column.
    This function can be deleted after running once if the CSV is already updated.
    """
    import os
    
    print(f"\n{'='*60}")
    print(f"Creating new CSV file with time column: {filename}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"File {filename} not found. Skipping update.")
        return
    
    # Create output filename
    base_name = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1]
    output_filename = f"{base_name}_with_time{extension}"
    
    # Check if output file already exists and has time column
    if os.path.exists(output_filename):
        try:
            df_check = pd.read_csv(output_filename, header=0, skiprows=[1])
            if 'time' in df_check.columns:
                print(f"Output file {output_filename} already exists with time column.")
                print(f"Using existing file: {output_filename}")
                return output_filename
        except:
            pass
    
    # Read CSV with first row as headers, skip the second row (metadata)
    df = pd.read_csv(filename, header=0, skiprows=[1])
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Data shape: {df.shape}")
    
    # Check if time column already exists
    if 'time' in df.columns:
        print("'time' column already exists in CSV. Creating copy anyway.")
        # Still create the new file for consistency
    
    # Add time column from X or Sequence
    if 'X' in df.columns:
        df['time'] = df['X'] * 0.00004
        print("Added 'time' column from X * 0.00004")
    elif 'Sequence' in df.columns:
        df['time'] = df['Sequence'] * 0.00004
        print("Added 'time' column from Sequence * 0.00004")
    else:
        print("Neither 'X' nor 'Sequence' column found. Skipping time column update.")
        return filename
    
    # Read the original file to preserve the metadata row
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        return filename
    
    # Reconstruct the file with metadata row
    print(f"\nWriting new file: {output_filename}...")
    
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
    
    # Write to new file
    try:
        with open(output_filename, 'w', newline='') as f:
            f.write(header_line + '\n')
            f.write(metadata_line + '\n')
            
            # Write data rows
            for idx, row in df.iterrows():
                row_values = [str(row[col]) for col in df.columns]
                f.write(','.join(row_values) + '\n')
        
        print(f"Successfully created {output_filename}")
        print(f"New columns: {df.columns.tolist()}")
        print(f"Time column range: {df['time'].min():.6f} to {df['time'].max():.6f}")
        print(f"\nOriginal file: {filename}")
        print(f"New file with time: {output_filename}")
    except Exception as e:
        print(f"Error writing file: {e}")
        raise
    
    print(f"{'='*60}\n")
    return output_filename

if __name__ == '__main__':
    original_filename = 'Combined_book4.csv'
    # Update CSV with time column before running main (creates new file) - this is really just a one time modificaiton, due to errors associated with the original 
    # data 'X' column being text rather than integers.
    # updated_filename = update_csv_with_time(original_filename)
    # # Use the updated filename if a new file was created, otherwise use original
    # if updated_filename and updated_filename != original_filename:
    #     filename_to_use = updated_filename
    # else:
    #     filename_to_use = original_filename
    filename_to_use = "Combined_book4_with_time.csv"
    
    # ============================================================================
    # MANUAL Y-AXIS LIMIT CONFIGURATION - Set these to override Y-axis limits
    # ============================================================================
    # Set MANUAL_Y_MAX to a number to hardcode the upper Y-axis limit for FFT plot
    # Set MANUAL_Y_MIN to a number to hardcode the lower Y-axis limit for FFT plot
    # Leave as None to use the values from the UI controls
    MANUAL_Y_MAX = 400  # Example: MANUAL_Y_MAX = 100.0  to truncate at 100
    MANUAL_Y_MIN = None  # Example: MANUAL_Y_MIN = 0.0    to set minimum to 0
    
    # ============================================================================
    # MANUAL FREQUENCY LIMIT CONFIGURATION - Set these to override frequency limits
    # ============================================================================
    # Set MANUAL_FREQ_START to a number to hardcode the start frequency for FFT plot
    # Set MANUAL_FREQ_END to a number to hardcode the end frequency for FFT plot
    # Leave as None to use the values from the UI controls
    MANUAL_FREQ_START = None  # Example: MANUAL_FREQ_START = 0.0  to start at 0
    MANUAL_FREQ_END = 30    # Example: MANUAL_FREQ_END = 100.0  to end at 100
    # ============================================================================
    
    main(filename_to_use, MANUAL_Y_MIN, MANUAL_Y_MAX, MANUAL_FREQ_START, MANUAL_FREQ_END)
