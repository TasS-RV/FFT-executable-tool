"""
This script is used to generate the cogging torque plot and FFT spectrum for the original and Syntec Motor data.

This was developed by Phill Phillippou, 2025 - and is used to generate the cogging torque plot and FFT spectrum data.
"""

from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk


from tkinter.filedialog import askopenfilename


# Define file paths
# file1 = "C:/Users/phil.philippou/source/repos/cogging/torque ripple.csv"
# file2 = "C:/Users/phil.philippou/source/repos/cogging/085_2U_1132001_original_rotor_and_stator.csv"

# file1 = "./085_2U_1132001_original_rotor_and_stator 1.csv"
# file2 = "./Syntec Motor.csv"
file_syntec = "./Syntec Motor.csv"


# Define resampling factor (adjust for resolution control)
resample_factor = 5  # Higher values = Lower resolution

# Read CSV while skipping the second row and selecting CH1, CH2, CH3
# df1 = pd.read_csv(file1, skiprows=[1], usecols=[0, 1, 2, 3])  # X, CH1 (Torque), CH2 (Sine), CH3 (Cosine)
# df2 = pd.read_csv(file2, skiprows=[1], usecols=[0, 1, 2, 3])  # X, CH1 (Torque), CH2 (Sine), CH3 (Cosine)
df_syntec = pd.read_csv(file_syntec, skiprows=[1], usecols=[0, 1, 2, 3])


# Convert columns to numeric
# df1 = df1.apply(pd.to_numeric, errors="coerce")
# df2 = df2.apply(pd.to_numeric, errors="coerce")
df_syntec = df_syntec.apply(pd.to_numeric, errors="coerce")

# Remove DC offset from CH2 and CH3
# df1["CH2"] -= df1["CH2"].mean()
# df1["CH3"] -= df1["CH3"].mean()
# df2["CH2"] -= df2["CH2"].mean()
# df2["CH3"] -= df2["CH3"].mean()
df_syntec["CH2"] -= df_syntec["CH2"].mean()
df_syntec["CH3"] -= df_syntec["CH3"].mean()

# Compute raw angles using arctan2
# df1["Theta (radians)"] = np.arctan2(df1["CH2"], df1["CH3"])
# df2["Theta (radians)"] = np.arctan2(df2["CH2"], df2["CH3"])
df_syntec["Theta (radians)"] = np.arctan2(df_syntec["CH2"], df_syntec["CH3"])
print (df_syntec)


# Unwrap phase to ensure continuous mechanical rotation
# df1["Unwrapped Theta (radians)"] = np.unwrap(df1["Theta (radians)"])
# df2["Unwrapped Theta (radians)"] = np.unwrap(df2["Theta (radians)"])
df_syntec["Unwrapped Theta (radians)"] = np.unwrap(df_syntec["Theta (radians)"])




# Convert to mechanical angle (5000 sine/cosine cycles per revolution)
# df1["Accumulated Theta (degrees)"] = (df1["Unwrapped Theta (radians)"] / (2 * np.pi)) * (360 / 5000)
# df2["Accumulated Theta (degrees)"] = (df2["Unwrapped Theta (radians)"] / (2 * np.pi)) * (360 / 5000)
df_syntec["Accumulated Theta (degrees)"] = (df_syntec["Unwrapped Theta (radians)"] / (2 * np.pi)) * (360 / 5000)
print (df_syntec)


# Determine theta range for Syntec Motor
# theta_min = max(np.min(df1["Accumulated Theta (degrees)"]), np.min(df2["Accumulated Theta (degrees)"]))
# theta_max = min(np.max(df1["Accumulated Theta (degrees)"]), np.max(df2["Accumulated Theta (degrees)"]))
# theta_range = theta_max - theta_min
theta_min = np.min(df_syntec["Accumulated Theta (degrees)"])
theta_max = np.max(df_syntec["Accumulated Theta (degrees)"])
theta_range = theta_max - theta_min
print (theta_range)

# Compute dynamic angle step based on number of points and resample factor
# num_points = min(len(df1), len(df2))
# mechanical_angle_step = (theta_range / num_points) * resample_factor  # Adjusted step size
num_points = len(df_syntec)
mechanical_angle_step = (theta_range / num_points) * resample_factor  # Adjusted step size
print (mechanical_angle_step)


# Generate new fixed theta grid
theta_fixed = np.arange(theta_min, theta_max, mechanical_angle_step)

# Interpolate CH1 at new theta values
# ch1_interp_1 = np.interp(theta_fixed, df1["Accumulated Theta (degrees)"], df1["CH1"])
# ch1_interp_2 = np.interp(theta_fixed, df2["Accumulated Theta (degrees)"], df2["CH1"])
ch1_interp_syntec = np.interp(theta_fixed, df_syntec["Accumulated Theta (degrees)"], df_syntec["CH1"])
print (ch1_interp_syntec)


# Compute new sampling rate (Fs) in Hz
Fs = 1 / mechanical_angle_step  
print (Fs)

# Perform FFT on interpolated CH1
# fft_ch1_interp_1 = np.fft.fft(ch1_interp_1)
# fft_ch1_interp_2 = np.fft.fft(ch1_interp_2)
fft_ch1_interp_syntec = np.fft.fft(ch1_interp_syntec)


# Remove DC component
# fft_ch1_interp_1[0] = 0
# fft_ch1_interp_2[0] = 0
fft_ch1_interp_syntec[0] = 0


# Compute frequency bins
freqs = np.fft.fftfreq(len(ch1_interp_syntec), d=1/Fs)

# Compute magnitude spectrum and normalize
# magnitude_interp_1 = np.abs(fft_ch1_interp_1) / len(ch1_interp_1)
# magnitude_interp_2 = np.abs(fft_ch1_interp_2) / len(ch1_interp_2)
magnitude_interp_syntec = np.abs(fft_ch1_interp_syntec) / len(ch1_interp_syntec)

# Define zoom range for visualization (up to 1 Hz)
#max_freq = 1  
max_freq = 0.2 # This is not the frequency directly - it is a scaling factor of what to truncate to
mask = (freqs >= 0) & (freqs <= max_freq)
freqs_zoomed = freqs[mask] *360
# magnitude_zoomed_1 = magnitude_interp_1[mask]
# magnitude_zoomed_2 = magnitude_interp_2[mask]
magnitude_zoomed_syntec = magnitude_interp_syntec[mask]


# Plot Cogging Torque vs Accumulated Theta (Syntec Motor only)
plt.figure(figsize=(12, 5))
# plt.plot(theta_fixed, ch1_interp_1, label="085_2U_1132001_original_rotor_and_stator: CH1 vs. Accumulated Theta", linestyle="-")
# plt.plot(theta_fixed, ch1_interp_2, label="Syntec Motor: CH1 vs. Accumulated Theta", linestyle="-")
plt.plot(theta_fixed, ch1_interp_syntec, label="Syntec Motor: CH1 vs. Accumulated Theta", linestyle="-")
plt.xlabel("Accumulated Theta (Degrees)")
plt.ylabel("Cogging Torque")
plt.title("Cogging Torque vs. Accumulated Theta")
plt.legend()
plt.grid()
plt.show()

# Plot FFT spectrum of Interpolated CH1 (Syntec Motor only)
plt.figure(figsize=(10, 5))
# plt.plot(freqs_zoomed, magnitude_zoomed_1, label="085_2U_1132001_original_rotor_and_stator: FFT of CH1", linestyle="-")
# plt.plot(freqs_zoomed, magnitude_zoomed_2, label="Syntec Motor: FFT of CH1", linestyle="-")
plt.plot(freqs_zoomed, magnitude_zoomed_syntec, label="Syntec Motor: FFT of CH1", linestyle="-")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (scaled)")
plt.title("FFT of Interpolated CH1 (DC Removed, Zoomed to 1 Hz)")
plt.legend()
plt.grid()
plt.show()

# Polar Plot of Cogging Torque vs Accumulated Theta (Syntec Motor only)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
# ax.plot(np.radians(theta_fixed), ch1_interp_1, label="085_2U_1132001_original_rotor_and_stator: CH1")
# ax.plot(np.radians(theta_fixed), ch1_interp_2, label="Syntec Motor: CH1")
ax.plot(np.radians(theta_fixed), ch1_interp_syntec, label="Syntec Motor: CH1")
ax.set_title("Polar Plot of Cogging Torque vs. Mechanical Angle")
ax.legend()
plt.show()

