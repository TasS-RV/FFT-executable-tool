"""
Test Data Generator for FFT Analysis Tool

This script generates CSV files with known signal characteristics for testing
the FFT analysis tool. Each test case includes:
- Superimposed sine waves at known frequencies
- Various noise characteristics (white noise, high/low frequency noise)
- Non-uniform sampling (optional)

Test cases are designed to validate:
- FFT frequency detection accuracy
- Filter effectiveness
- Interpolation behavior
- Subsampling effects

Files are saved as CSV format and will NOT overwrite existing files.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path


# Configuration
OUTPUT_DIR = Path(__file__).parent  # Save in same directory as this script
SAMPLE_RATE = 1000.0  # Hz - base sampling rate
DURATION = 10.0  # seconds
N_POINTS = int(SAMPLE_RATE * DURATION)  # Total number of points


def generate_time_base(sample_rate=SAMPLE_RATE, duration=DURATION, uniform=True, jitter=0.0):
    """
    Generate time base (x-axis) for signals.
    
    Args:
        sample_rate: Base sampling rate in Hz
        duration: Duration in seconds
        uniform: If True, uniform spacing; if False, add jitter
        jitter: Amount of random jitter to add (as fraction of dt)
    
    Returns:
        time array
    """
    dt = 1.0 / sample_rate
    if uniform:
        t = np.linspace(0, duration, N_POINTS)
    else:
        # Add random jitter to simulate non-uniform sampling
        t_base = np.linspace(0, duration, N_POINTS)
        jitter_amount = dt * jitter * np.random.randn(N_POINTS)
        t = t_base + jitter_amount
        t = np.sort(t)  # Ensure monotonic
        t = t - t[0]  # Start at 0
    return t


def generate_signal(t, frequencies, amplitudes, phases=None, noise_level=0.0, 
                   noise_type='white', high_freq_noise=0.0, low_freq_noise=0.0):
    """
    Generate signal with superimposed sine waves and noise.
    
    Args:
        t: Time array
        frequencies: List of frequencies (Hz) for sine waves
        amplitudes: List of amplitudes for each frequency
        phases: List of phases (radians), defaults to random
        noise_level: Amplitude of white noise
        noise_type: 'white', 'pink', or 'brown'
        high_freq_noise: Additional high-frequency noise amplitude
        low_freq_noise: Additional low-frequency noise amplitude
    
    Returns:
        Signal array
    """
    signal = np.zeros_like(t)
    
    # Add sine waves
    if phases is None:
        phases = np.random.uniform(0, 2*np.pi, len(frequencies))
    
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Add white noise
    if noise_level > 0:
        signal += noise_level * np.random.randn(len(t))
    
    # Add high-frequency noise (simulated as filtered white noise)
    if high_freq_noise > 0:
        # High-frequency noise: random high-frequency components
        # Simulate high-frequency content by adding rapid oscillations
        hf_noise = high_freq_noise * np.random.randn(len(t))
        # Apply high-pass characteristic (simple differentiation, then pad)
        hf_diff = np.diff(hf_noise)
        hf_noise = np.concatenate([[hf_diff[0]], hf_diff])  # Maintain length
        signal += hf_noise
    
    # Add low-frequency noise (drift)
    if low_freq_noise > 0:
        # Low-frequency noise: slow random walk
        lf_noise = np.cumsum(low_freq_noise * np.random.randn(len(t)) * (t[1] - t[0]))
        signal += lf_noise
    
    return signal


def save_test_data(filename, t, signal, description):
    """
    Save test data to CSV file if it doesn't already exist.
    
    Args:
        filename: Output filename
        t: Time array (x-axis)
        signal: Signal array (y-axis)
        description: Description of the test case
    """
    filepath = OUTPUT_DIR / filename
    
    # Check if file already exists
    if filepath.exists():
        print(f"  ⚠ Skipping {filename} - file already exists")
        return False
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': t,
        'signal': signal
    })
    
    # Save to CSV
    try:
        df.to_csv(filepath, index=False)
        print(f"  ✓ Created {filename}")
        print(f"    Description: {description}")
        return True
    except Exception as e:
        print(f"  ✗ Error creating {filename}: {e}")
        return False


def main():
    """Generate all test data files."""
    print("=" * 70)
    print("FFT Test Data Generator")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Total points: {N_POINTS}")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # ========================================================================
    # TEST CASE 1: Clean dual-frequency signal (no noise)
    # ========================================================================
    print("\n[Test Case 1] Clean dual-frequency signal")
    t1 = generate_time_base(uniform=True)
    signal1 = generate_signal(
        t1,
        frequencies=[10.0, 50.0],  # Two distinct frequencies
        amplitudes=[1.0, 0.5],
        noise_level=0.0
    )
    save_test_data(
        "test_01_clean_dual_freq.csv",
        t1, signal1,
        "Clean signal with two sine waves at 10 Hz and 50 Hz. No noise. "
        "Tests basic FFT frequency detection accuracy."
    )
    
    # ========================================================================
    # TEST CASE 2: Multiple frequencies with white noise
    # ========================================================================
    print("\n[Test Case 2] Multiple frequencies with white noise")
    t2 = generate_time_base(uniform=True)
    signal2 = generate_signal(
        t2,
        frequencies=[5.0, 25.0, 100.0, 150.0],  # Multiple frequencies
        amplitudes=[2.0, 1.0, 0.8, 0.6],
        noise_level=0.2  # Moderate white noise
    )
    save_test_data(
        "test_02_multi_freq_white_noise.csv",
        t2, signal2,
        "Signal with 4 sine waves (5, 25, 100, 150 Hz) plus white noise. "
        "Tests FFT's ability to detect multiple frequencies in noisy data."
    )
    
    # ========================================================================
    # TEST CASE 3: High-frequency noise contamination
    # ========================================================================
    print("\n[Test Case 3] High-frequency noise contamination")
    t3 = generate_time_base(uniform=True)
    signal3 = generate_signal(
        t3,
        frequencies=[20.0, 60.0],
        amplitudes=[1.5, 1.0],
        noise_level=0.1,
        high_freq_noise=0.5  # Strong high-frequency noise
    )
    save_test_data(
        "test_03_high_freq_noise.csv",
        t3, signal3,
        f"Signal with 20 Hz and 60 Hz components plus strong high-frequency noise at 10% amplitude modulation."
        "Tests low-pass filter effectiveness."
    )
    
    # ========================================================================
    # TEST CASE 4: Low-frequency drift/noise
    # ========================================================================
    print("\n[Test Case 4] Low-frequency drift/noise")
    t4 = generate_time_base(uniform=True)
    signal4 = generate_signal(
        t4,
        frequencies=[30.0, 80.0],
        amplitudes=[1.0, 0.7],
        noise_level=0.15,
        low_freq_noise=0.3  # Low-frequency drift
    )
    save_test_data(
        "test_04_low_freq_drift.csv",
        t4, signal4,
        "Signal with 30 Hz and 80 Hz components plus low-frequency drift. "
        "Tests high-pass filter effectiveness for removing DC and drift."
    )
    
    # ========================================================================
    # TEST CASE 5: Mains frequency (50 Hz) interference
    # ========================================================================
    print("\n[Test Case 5] Mains frequency (50 Hz) interference")
    t5 = generate_time_base(uniform=True)
    signal5 = generate_signal(
        t5,
        frequencies=[15.0, 50.0, 75.0],  # 50 Hz is mains interference
        amplitudes=[1.0, 2.0, 0.8],  # Strong 50 Hz component
        noise_level=0.2
    )
    save_test_data(
        "test_05_mains_interference.csv",
        t5, signal5,
        "Signal with 15 Hz, 75 Hz components plus strong 50 Hz mains interference. "
        "Tests notch filter effectiveness at removing specific frequencies."
    )
    
    # ========================================================================
    # TEST CASE 6: Multiple notch frequencies (50 Hz and 100 Hz)
    # ========================================================================
    print("\n[Test Case 6] Multiple notch frequencies")
    t6 = generate_time_base(uniform=True)
    signal6 = generate_signal(
        t6,
        frequencies=[25.0, 50.0, 100.0, 125.0],
        amplitudes=[1.5, 1.8, 1.6, 1.2],  # Strong interference at 50 and 100 Hz
        noise_level=0.15
    )
    save_test_data(
        "test_06_multiple_notch_freqs.csv",
        t6, signal6,
        "Signal with desired frequencies (25, 125 Hz) plus interference at 50 and 100 Hz. "
        "Tests multiple notch filters simultaneously."
    )
    
    # ========================================================================
    # TEST CASE 7: Non-uniform sampling (with jitter)
    # ========================================================================
    print("\n[Test Case 7] Non-uniform sampling")
    t7 = generate_time_base(uniform=False, jitter=0.1)  # 10% jitter
    signal7 = generate_signal(
        t7,
        frequencies=[12.0, 45.0],
        amplitudes=[1.0, 0.6],
        noise_level=0.1
    )
    save_test_data(
        "test_07_nonuniform_sampling.csv",
        t7, signal7,
        "Signal with non-uniform time spacing (10% jitter). "
        "Tests interpolation accuracy and FFT behavior with irregular sampling."
    )
    
    # ========================================================================
    # TEST CASE 8: Very noisy signal (low SNR)
    # ========================================================================
    print("\n[Test Case 8] Very noisy signal (low SNR)")
    t8 = generate_time_base(uniform=True)
    signal8 = generate_signal(
        t8,
        frequencies=[35.0, 90.0],
        amplitudes=[0.5, 0.4],  # Weak signals
        noise_level=1.0,  # Strong noise (SNR ~ 0.5)
        high_freq_noise=0.3,
        low_freq_noise=0.2
    )
    save_test_data(
        "test_08_low_snr.csv",
        t8, signal8,
        "Weak signals (35, 90 Hz) buried in strong noise. "
        "Tests filter pipeline effectiveness for low signal-to-noise ratio."
    )
    
    # ========================================================================
    # TEST CASE 9: Close frequency pairs (frequency resolution test)
    # ========================================================================
    print("\n[Test Case 9] Close frequency pairs")
    t9 = generate_time_base(uniform=True)
    signal9 = generate_signal(
        t9,
        frequencies=[40.0, 40.5, 95.0, 96.0],  # Close frequencies
        amplitudes=[1.0, 0.9, 0.8, 0.7],
        noise_level=0.1
    )
    save_test_data(
        "test_09_close_frequencies.csv",
        t9, signal9,
        "Signal with closely spaced frequencies (40.0/40.5 Hz, 95.0/96.0 Hz). "
        "Tests FFT frequency resolution and ability to distinguish nearby peaks."
    )
    
    # ========================================================================
    # TEST CASE 10: Wide frequency range (low to high)
    # ========================================================================
    print("\n[Test Case 10] Wide frequency range")
    t10 = generate_time_base(uniform=True)
    signal10 = generate_signal(
        t10,
        frequencies=[2.0, 20.0, 100.0, 200.0],  # Wide range
        amplitudes=[1.2, 1.0, 0.9, 0.7],
        noise_level=0.15
    )
    save_test_data(
        "test_10_wide_freq_range.csv",
        t10, signal10,
        "Signal with frequencies spanning 2-200 Hz. "
        "Tests FFT performance across wide frequency range and filter bandwidth selection."
    )
    
    # ========================================================================
    # TEST CASE 11: Spike contamination (for median filter test)
    # ========================================================================
    print("\n[Test Case 11] Spike contamination")
    t11 = generate_time_base(uniform=True)
    signal11 = generate_signal(
        t11,
        frequencies=[30.0, 70.0],
        amplitudes=[1.0, 0.8],
        noise_level=0.1
    )
    # Add random spikes
    n_spikes = 50
    spike_indices = np.random.choice(len(signal11), n_spikes, replace=False)
    signal11[spike_indices] += np.random.choice([-1, 1], n_spikes) * 5.0  # Large spikes
    save_test_data(
        "test_11_spikes.csv",
        t11, signal11,
        "Signal with 30 Hz and 70 Hz components plus random large spikes. "
        "Tests median filter effectiveness for removing outliers/spikes."
    )
    
    # ========================================================================
    # TEST CASE 12: All noise types combined
    # ========================================================================
    print("\n[Test Case 12] All noise types combined")
    t12 = generate_time_base(uniform=True)
    signal12 = generate_signal(
        t12,
        frequencies=[18.0, 55.0, 120.0],
        amplitudes=[1.5, 1.2, 1.0],
        noise_level=0.3,
        high_freq_noise=0.4,
        low_freq_noise=0.3
    )
    # Add some spikes
    spike_indices = np.random.choice(len(signal12), 30, replace=False)
    signal12[spike_indices] += np.random.choice([-1, 1], 30) * 3.0
    save_test_data(
        "test_12_all_noise_types.csv",
        t12, signal12,
        "Signal with multiple frequencies plus white noise, high-freq noise, "
        "low-freq drift, and spikes. Comprehensive filter pipeline test."
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Test data generation complete!")
    print("=" * 70)
    print("\nTest files created in:", OUTPUT_DIR)
    print("\nExpected FFT peaks for each test case:")
    print("  Test 01: 10 Hz, 50 Hz")
    print("  Test 02: 5 Hz, 25 Hz, 100 Hz, 150 Hz")
    print("  Test 03: 20 Hz, 60 Hz (high-freq noise present)")
    print("  Test 04: 30 Hz, 80 Hz (low-freq drift present)")
    print("  Test 05: 15 Hz, 50 Hz, 75 Hz (50 Hz is interference)")
    print("  Test 06: 25 Hz, 50 Hz, 100 Hz, 125 Hz (50 & 100 Hz interference)")
    print("  Test 07: 12 Hz, 45 Hz (non-uniform sampling)")
    print("  Test 08: 35 Hz, 90 Hz (very noisy)")
    print("  Test 09: 40 Hz, 40.5 Hz, 95 Hz, 96 Hz (close frequencies)")
    print("  Test 10: 2 Hz, 20 Hz, 100 Hz, 200 Hz (wide range)")
    print("  Test 11: 30 Hz, 70 Hz (with spikes)")
    print("  Test 12: 18 Hz, 55 Hz, 120 Hz (all noise types)")
    print()


if __name__ == "__main__":
    main()

