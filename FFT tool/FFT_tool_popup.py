import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import matplotlib.gridspec as gridspec
import tkinter as tk
from tkinter import ttk

def read_csv_data(filename):
    """Read the CSV file and extract all data, adding time column if needed."""
    # Read CSV with first row as headers, skip the second row (metadata with units)
    df_full = pd.read_csv(filename, header=0, skiprows=[1])
    
    # Get available columns
    available_columns = df_full.columns.tolist()
    print(f"Available columns: {available_columns}")
    
    # Add time column if X or Sequence column exists
    if 'X' in available_columns:
        df_full['time'] = df_full['X'] * 0.00004
        print("Added 'time' column from X * 0.00004")
    elif 'Sequence' in available_columns:
        df_full['time'] = df_full['Sequence'] * 0.00004
        print("Added 'time' column from Sequence * 0.00004")
    
    # Update available columns after adding time
    available_columns = df_full.columns.tolist()
    
    # For backward compatibility, try to get position and CH4 if they exist
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
    """Compute FFT of signal with given step size dx and optional frequency limits."""
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

def update_plots(dx, df, x_col, y_col, base_increment, freq_start, freq_end, 
                 ax_fft, ax_data, line_fft, line_data):
    """Update FFT and data plots."""
    try:
        # Get data from selected columns
        x_data = df[x_col].values
        y_data = df[y_col].values
        
        # Calculate dx if x_col is position-like
        if base_increment is None:
            # Calculate from x_data differences
            x_diff = np.diff(x_data)
            unique_diffs = np.unique(x_diff)
            current_dx = np.min(unique_diffs[unique_diffs > 0])
        else:
            current_dx = dx
        
        # Compute FFT
        freq, fft_mag = compute_fft(y_data, current_dx, freq_start, freq_end)
        
        # Update FFT plot
        line_fft.set_data(freq, fft_mag)
        
        # Update FFT axis limits
        if len(freq) > 0:
            x_min = freq[0]
            x_max = freq[-1]
            
            if freq_start is not None:
                x_min = max(x_min, freq_start)
            if freq_end is not None:
                x_max = min(x_max, freq_end)
            
            ax_fft.set_xlim(x_min, x_max)
            ax_fft.set_ylim(0, np.max(fft_mag) * 1.1)
        else:
            ax_fft.set_xlim(0, 1)
            ax_fft.set_ylim(0, 1)
        
        # Update data plot
        line_data.set_data(x_data, y_data)
        
        # Update data axis limits
        ax_data.set_xlim(np.min(x_data), np.max(x_data))
        ax_data.set_ylim(np.min(y_data) * 1.05, np.max(y_data) * 1.05)
        
        # Update labels
        ax_fft.set_xlabel(f'Spatial Frequency (1/{x_col})', fontsize=10)
        ax_fft.set_ylabel(f'FFT Magnitude', fontsize=10)
        ax_data.set_xlabel(f'{x_col}', fontsize=10)
        ax_data.set_ylabel(f'{y_col}', fontsize=10)
        
        plt.draw()
    except Exception as e:
        print(f"Error updating plots: {e}")

def create_control_window(df, available_columns, default_x_col, default_y_col, 
                         base_increment, initial_dx, callback):
    """Create a tkinter popup window with controls."""
    root = tk.Tk()
    root.title("FFT Control Panel")
    root.geometry("400x500")
    
    # Variables
    x_col_var = tk.StringVar(value=default_x_col)
    y_col_var = tk.StringVar(value=default_y_col)
    freq_start_var = tk.StringVar(value="")
    freq_end_var = tk.StringVar(value="")
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
        # Get values
        x_col = x_col_var.get()
        y_col = y_col_var.get()
        multiplier = slider_var.get()
        new_dx = (base_increment * multiplier) if base_increment is not None else (initial_dx * multiplier)
        
        # Parse frequency inputs
        freq_start_val = None
        freq_end_val = None
        
        freq_start_str = freq_start_var.get().strip()
        if freq_start_str:
            try:
                freq_start_val = float(freq_start_str)
            except ValueError:
                print(f"Invalid start frequency: {freq_start_str}")
        
        freq_end_str = freq_end_var.get().strip()
        if freq_end_str:
            try:
                freq_end_val = float(freq_end_str)
            except ValueError:
                print(f"Invalid end frequency: {freq_end_str}")
        
        freq_start_to_use = freq_start_val if freq_start_val is not None else 0.0
        freq_end_to_use = freq_end_val
        
        # Call the callback with parameters
        callback(x_col, y_col, new_dx, freq_start_to_use, freq_end_to_use)
    
    run_button = tk.Button(root, text="Run FFT", command=run_fft, 
                          bg="lightblue", font=("Arial", 12, "bold"),
                          width=20, height=2)
    run_button.pack(pady=20)
    
    # Close button
    close_button = tk.Button(root, text="Close", command=root.quit,
                             font=("Arial", 10), width=15)
    close_button.pack(pady=5)
    
    return root, x_col_var, y_col_var, freq_start_var, freq_end_var, slider_var

def main():
    # Read data
    global filename
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
    current_x_col = [default_x_col]
    current_y_col = [default_y_col]
    
    # Create matplotlib figure
    fig = None
    ax_fft = None
    ax_data = None
    line_fft = None
    line_data = None
    
    def create_plots(x_col, y_col, dx, freq_start, freq_end):
        nonlocal fig, ax_fft, ax_data, line_fft, line_data
        
        # Close existing figure if it exists
        if fig is not None:
            plt.close(fig)
        
        # Create new figure
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        ax_fft = fig.add_subplot(gs[0])
        ax_data = fig.add_subplot(gs[1])
        
        # Get data
        x_data = df[x_col].values
        y_data = df[y_col].values
        
        # Compute FFT
        freq, fft_mag = compute_fft(y_data, dx, freq_start, freq_end)
        
        # Plot FFT
        line_fft, = ax_fft.plot(freq, fft_mag, 'r-', linewidth=1.5)
        
        # Plot data
        line_data, = ax_data.plot(x_data, y_data, 'b-', linewidth=0.5, alpha=0.7)
        
        # Configure axes
        ax_fft.set_title('FFT Plot', fontsize=12, fontweight='bold')
        ax_fft.set_xlabel(f'Spatial Frequency (1/{x_col})', fontsize=10)
        ax_fft.set_ylabel('Magnitude', fontsize=10)
        ax_fft.grid(True, alpha=0.3)
        
        ax_data.set_title('Data Plot', fontsize=12, fontweight='bold')
        ax_data.set_xlabel(f'{x_col}', fontsize=10)
        ax_data.set_ylabel(f'{y_col}', fontsize=10)
        ax_data.grid(True, alpha=0.3)
        
        plt.suptitle('FFT Analysis Tool', fontsize=14, fontweight='bold')
        plt.show(block=False)
    
    def run_fft_callback(x_col, y_col, dx, freq_start, freq_end):
        """Callback function called when Run FFT is pressed."""
        print(f"Running FFT: X={x_col}, Y={y_col}, dx={dx:.6f}, "
              f"freq_start={freq_start}, freq_end={freq_end}")
        create_plots(x_col, y_col, dx, freq_start, freq_end)
    
    # Create initial plots
    create_plots(default_x_col, default_y_col, initial_dx, 0.0, None)
    
    # Create control window
    control_window, x_var, y_var, freq_start_var, freq_end_var, slider_var = \
        create_control_window(df, available_columns, default_x_col, default_y_col,
                            base_increment, initial_dx, run_fft_callback)
    
    # Run tkinter main loop
    control_window.mainloop()
    
    # Close matplotlib when control window closes
    if fig is not None:
        plt.close('all')

if __name__ == '__main__':
    filename = 'Combined_book4.csv'
    main()

