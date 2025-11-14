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
from pathlib import Path


# Configuration
OUTPUT_DIR = Path(__file__).parent  # Save in same directory as this script
SAMPLE_RATE = 1000.0  # Hz - base sampling rate
DURATION = 10.0  # seconds
N_POINTS = int(SAMPLE_RATE * DURATION)  # Total number of points


def generate_time_base(sample_rate=SAMPLE_RATE, duration=DURATION, uniform=True, jitter=0.0):
    """
    Generate time base (x-axis) for signals.
    """
    dt = 1.0 / sample_rate
    if uniform:
        t = np.linspace(0, duration, N_POINTS)
    else:
        t_base = np.linspace(0, duration, N_POINTS)
        jitter_amount = dt * jitter * np.random.randn(N_POINTS)
        t = t_base + jitter_amount
        t = np.sort(t)
        t = t - t[0]
    return t


def generate_signal(t, frequencies, amplitudes, phases=None, noise_level=0.0,
                    noise_type='white', high_freq_noise=0.0, low_freq_noise=0.0):
    """
    Generate signal with superimposed sine waves and noise.
    """
    signal = np.zeros_like(t)

    if phases is None:
        phases = np.random.uniform(0, 2 * np.pi, len(frequencies))

    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        signal += amp * np.sin(2 * np.pi * freq * t + phase)

    if noise_level > 0:
        base_noise = np.random.randn(len(t))
        if noise_type == 'white':
            colored = base_noise
        elif noise_type == 'pink':
            freqs = np.fft.rfftfreq(len(t), d=(t[1] - t[0]))
            spec = np.fft.rfft(base_noise)
            spec /= np.sqrt(freqs + 1e-6)
            colored = np.fft.irfft(spec, n=len(t))
        elif noise_type == 'brown':
            colored = np.cumsum(base_noise)
        else:
            colored = base_noise
        colored -= np.mean(colored)
        colored /= np.std(colored) + 1e-9
        signal += noise_level * colored

    if len(t) > 1:
        dt = np.median(np.diff(t))
        fs_est = 1.0 / dt if dt > 0 else 1.0
    else:
        dt = 1.0
        fs_est = 1.0

    if high_freq_noise > 0:
        n = len(t)
        freqs = np.fft.rfftfreq(n, d=dt)
        spec = (np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs)))
        high_mask = freqs >= 0.25 * fs_est
        spec *= high_mask * (1.0 + (freqs / (fs_est / 2.0))**1.5)
        hf = np.fft.irfft(spec, n=n)
        hf = hf - np.mean(hf)
        hf = hf / (np.std(hf) + 1e-9)
        signal += high_freq_noise * 4.0 * hf

    if low_freq_noise > 0:
        n = len(t)
        freqs = np.fft.rfftfreq(n, d=dt)
        spec = (np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs)))
        low_mask = freqs <= 0.1 * fs_est
        spec *= low_mask * (1.0 / (freqs + 1e-6))
        lf = np.fft.irfft(spec, n=n)
        lf = lf - np.mean(lf)
        lf = lf / (np.std(lf) + 1e-9)
        random_walk = np.cumsum(np.random.randn(n))
        random_walk = random_walk / (np.std(random_walk) + 1e-9)
        lf += random_walk
        signal += low_freq_noise * 6.0 * lf

    return signal


def save_test_data(filename, t, signal, description):
    """
    Save test data to CSV file if it doesn't already exist.
    """
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        print(f"  ⚠ Skipping {filename} - file already exists")
        return False

    df = pd.DataFrame({
        'time': t,
        'signal': signal
    })

    try:
        df.to_csv(filepath, index=False)
        print(f"  ✓ Created {filename}")
        print(f"    Description: {description}")
        return True
    except Exception as e:
        print(f"  ✗ Error creating {filename}: {e}")
        return False


def main():
    print("=" * 70)
    print("FFT Test Data Generator")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Total points: {N_POINTS}")
    print()

    np.random.seed(42)

    # TEST CASE 1
    print("\n[Test Case 1] Clean dual-frequency signal")
    t1 = generate_time_base(uniform=True)
    signal1 = generate_signal(
        t1,
        frequencies=[10.0, 50.0],
        amplitudes=[1.0, 0.5],
        noise_level=0.0
    )
    save_test_data(
        "test_01_clean_dual_freq.csv",
        t1, signal1,
        "Clean signal with two sine waves at 10 Hz and 50 Hz. No noise."
    )

    # TEST CASE 2
    print("\n[Test Case 2] Multiple frequencies with white noise")
    t2 = generate_time_base(uniform=True)
    signal2 = generate_signal(
        t2,
        frequencies=[5.0, 25.0, 100.0, 150.0],
        amplitudes=[2.0, 1.0, 0.8, 0.6],
        noise_level=0.2
    )
    save_test_data(
        "test_02_multi_freq_white_noise.csv",
        t2, signal2,
        "Signal with multiple sine waves plus white noise."
    )

    # TEST CASE 3
    print("\n[Test Case 3] High-frequency noise contamination")
    t3 = generate_time_base(uniform=True)
    signal3 = generate_signal(
        t3,
        frequencies=[20.0, 60.0],
        amplitudes=[1.2, 0.8],
        noise_level=0.4,
        high_freq_noise=1.2
    )
    save_test_data(
        "test_03_high_freq_noise.csv",
        t3, signal3,
        "Signal with 20 Hz and 60 Hz components plus strong high-frequency noise."
    )

    # TEST CASE 4
    print("\n[Test Case 4] Low-frequency drift/noise")
    t4 = generate_time_base(uniform=True)
    signal4 = generate_signal(
        t4,
        frequencies=[30.0, 80.0],
        amplitudes=[0.9, 0.6],
        noise_level=0.25,
        low_freq_noise=0.7
    )
    save_test_data(
        "test_04_low_freq_drift.csv",
        t4, signal4,
        "Signal with 30 Hz and 80 Hz components plus low-frequency drift."
    )

    # TEST CASE 5
    print("\n[Test Case 5] Mains frequency (50 Hz) interference")
    t5 = generate_time_base(uniform=True)
    signal5 = generate_signal(
        t5,
        frequencies=[15.0, 50.0, 75.0],
        amplitudes=[1.0, 2.0, 0.8],
        noise_level=0.2
    )
    save_test_data(
        "test_05_mains_interference.csv",
        t5, signal5,
        "Signal with 50 Hz mains interference."
    )

    # TEST CASE 6
    print("\n[Test Case 6] Multiple notch frequencies")
    t6 = generate_time_base(uniform=True)
    signal6 = generate_signal(
        t6,
        frequencies=[25.0, 50.0, 100.0, 125.0],
        amplitudes=[1.5, 1.8, 1.6, 1.2],
        noise_level=0.15
    )
    save_test_data(
        "test_06_multiple_notch_freqs.csv",
        t6, signal6,
        "Signal with desired frequencies plus interference at 50 and 100 Hz."
    )

    # TEST CASE 7
    print("\n[Test Case 7] Non-uniform sampling")
    t7 = generate_time_base(uniform=False, jitter=0.1)
    signal7 = generate_signal(
        t7,
        frequencies=[12.0, 45.0],
        amplitudes=[1.0, 0.6],
        noise_level=0.1
    )
    save_test_data(
        "test_07_nonuniform_sampling.csv",
        t7, signal7,
        "Signal with non-uniform time spacing."
    )

    # TEST CASE 8
    print("\n[Test Case 8] Very noisy signal (low SNR)")
    t8 = generate_time_base(uniform=True)
    signal8 = generate_signal(
        t8,
        frequencies=[35.0, 90.0],
        amplitudes=[0.35, 0.3],
        noise_level=1.5,
        high_freq_noise=0.55,
        low_freq_noise=0.45
    )
    save_test_data(
        "test_08_low_snr.csv",
        t8, signal8,
        "Weak signals buried in strong noise."
    )

    # TEST CASE 9
    print("\n[Test Case 9] Close frequency pairs")
    t9 = generate_time_base(uniform=True)
    signal9 = generate_signal(
        t9,
        frequencies=[40.0, 40.5, 95.0, 96.0],
        amplitudes=[1.0, 0.9, 0.8, 0.7],
        noise_level=0.1
    )
    save_test_data(
        "test_09_close_frequencies.csv",
        t9, signal9,
        "Signal with closely spaced frequencies."
    )

    # TEST CASE 10
    print("\n[Test Case 10] Wide frequency range")
    t10 = generate_time_base(uniform=True)
    signal10 = generate_signal(
        t10,
        frequencies=[2.0, 20.0, 100.0, 200.0],
        amplitudes=[1.2, 1.0, 0.9, 0.7],
        noise_level=0.15
    )
    save_test_data(
        "test_10_wide_freq_range.csv",
        t10, signal10,
        "Signal with frequencies spanning 2-200 Hz."
    )

    # TEST CASE 11
    print("\n[Test Case 11] Spike contamination")
    t11 = generate_time_base(uniform=True)
    signal11 = generate_signal(
        t11,
        frequencies=[30.0, 70.0],
        amplitudes=[1.0, 0.8],
        noise_level=0.1
    )
    n_spikes = 50
    spike_indices = np.random.choice(len(signal11), n_spikes, replace=False)
    signal11[spike_indices] += np.random.choice([-1, 1], n_spikes) * 5.0
    save_test_data(
        "test_11_spikes.csv",
        t11, signal11,
        "Signal with random large spikes."
    )

    # TEST CASE 12
    print("\n[Test Case 12] All noise types combined (new)")
    t12 = generate_time_base(uniform=True)


    """
    There are a lot more signal additions here, rather than just the base signal Instead of typical random noise generation, whcih is the simpler
    method used in the previous functions above, we are using a more complex method to generate the noise. It is more realistic, with a very low SNR.
    
    The noise is generated by adding a broadband noise, a colored noise, a random walk noise, and a dense harmonic noise.
    """

    base_signal12 = np.zeros_like(t12)
    freqs12 = [18.0, 55.0, 120.0]
    amps12 = [1.5, 1.2, 1.0]
    for f, a in zip(freqs12, amps12):
        base_signal12 += a * np.sin(2 * np.pi * f * t12 + np.random.uniform(0, 2 * np.pi))

    broadband_noise = np.random.randn(len(t12))
    broadband_noise -= np.mean(broadband_noise)
    broadband_noise /= np.std(broadband_noise) + 1e-9

    colored_spec = np.fft.rfft(np.random.randn(len(t12)))
    freqs = np.fft.rfftfreq(len(t12), d=(t12[1] - t12[0]))
    shaped_spec = colored_spec * (1.0 / (freqs + 1e-6))
    ultra_low = np.fft.irfft(shaped_spec, n=len(t12))
    ultra_low -= np.mean(ultra_low)
    ultra_low /= np.std(ultra_low) + 1e-9

    n_harmonics = 20
    dense_harmonics = np.zeros_like(t12)
    for _ in range(n_harmonics):
        freq = np.random.uniform(5.0, 500.0)
        amp = np.random.uniform(0.5, 3.0)
        phase = np.random.uniform(0, 2 * np.pi)
        dense_harmonics += amp * np.sin(2 * np.pi * freq * t12 + phase)

    random_walk = np.cumsum(np.random.randn(len(t12)))
    random_walk -= np.mean(random_walk)
    random_walk /= np.std(random_walk) + 1e-9

    aggressive_noise = (
        6.0 * broadband_noise +
        8.0 * ultra_low +
        4.5 * random_walk +
        dense_harmonics
    )

    signal12 = base_signal12 + aggressive_noise

    large_spikes = np.zeros_like(signal12)
    spike_indices = np.random.choice(len(signal12), 80, replace=False)
    large_spikes[spike_indices] = np.random.choice([-1, 1], size=80) * np.random.uniform(5, 15, size=80)
    signal12 += large_spikes

    save_test_data(
        "test_12_all_noise_types.csv",
        t12, signal12,
        "Extremely noisy signal combining broadband, colored, harmonic, and drift noise with spikes."
    )

    print("\n" + "=" * 70)
    print("Test data generation complete!")
    print("=" * 70)
    print("\nTest files created in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

