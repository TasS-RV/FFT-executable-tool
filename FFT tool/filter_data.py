import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, medfilt, detrend, savgol_filter

def estimate_fs_from_x(x):
    """
    Estimate the effective sampling frequency (Hz) from arbitrary x data.
    Uses median step spacing to be robust to noise or jitter.
    """
    dx = np.diff(x)
    dx = dx[np.isfinite(dx) & (dx > 0)]
    if len(dx) == 0:
        raise ValueError("Cannot estimate sampling interval — invalid x spacing.")
    dx_med = np.median(dx)
    return 1.0 / dx_med

def butter_sos(lowcut=None, highcut=None, fs=1.0, order=4):
    nyq = 0.5 * fs
    # Ensure valid normalized frequencies
    if lowcut is not None:
        low = max(lowcut / nyq, 1e-6)
    else:
        low = None
    if highcut is not None:
        high = min(highcut / nyq, 0.9999)
    else:
        high = None

    if low and high:
        if low >= high:
            raise ValueError(f"Invalid band: low={lowcut}Hz ≥ high={highcut}Hz for fs={fs}Hz")
        sos = butter(order, [low, high], btype="band", output="sos")
    elif high:
        sos = butter(order, high, btype="low", output="sos")
    elif low:
        sos = butter(order, low, btype="high", output="sos")
    else:
        raise ValueError("Specify lowcut and/or highcut for Butterworth filter.")
    return sos



def apply_sos_filter(y, sos):
    """Zero-phase filter using sosfiltfilt for stability."""
    from scipy.signal import sosfiltfilt
    return sosfiltfilt(sos, y)


def apply_notch(y, f0, fs, Q=30):
    """Narrowband notch filter centered at f0 Hz."""
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(f0 / (0.5 * fs), Q)
    return filtfilt(b, a, y)


def preprocess_signal(x, y,
                      do_median=True, median_k=5,
                      do_hp=True, hp_cut=0.1, hp_order=3,
                      do_notch=False, notch_freqs=None, notch_Q=30,
                      do_lp=True, lp_cut=250.0, lp_order=4,
                      do_savgol=False, sav_window=11, polyorder=3):
    """
    Full preprocessing pipeline for arbitrary (x, y) data.
    Filters are applied in the time domain (not on interpolated uniform data).
    Automatically estimates effective sampling frequency fs from x.
    """

    # 1️⃣ Estimate sampling frequency from x
    fs = estimate_fs_from_x(x)
    y_filt = np.copy(y)

    # 2️⃣ Median filter to remove spikes
    if do_median:
        y_filt = medfilt(y_filt, kernel_size=median_k)

    # 3️⃣ High-pass to remove drift / DC
    if do_hp:
        sos_hp = butter_sos(lowcut=hp_cut, highcut=None, fs=fs, order=hp_order)
        y_filt = apply_sos_filter(y_filt, sos_hp)

    # 4️⃣ Notch filters for mains or other narrowband noise
    if do_notch and notch_freqs is not None:
        for f0 in notch_freqs:
            y_filt = apply_notch(y_filt, f0=f0, fs=fs, Q=notch_Q)

    # 5️⃣ Low-pass to remove high-frequency noise
    if do_lp:
        sos_lp = butter_sos(lowcut=None, highcut=lp_cut, fs=fs, order=lp_order)
        y_filt = apply_sos_filter(y_filt, sos_lp)

    # 6️⃣ Optional Savitzky–Golay smoothing
    if do_savgol:
        if sav_window % 2 == 0:
            sav_window += 1
        sav_window = min(sav_window, len(y_filt) - 1)
        y_filt = savgol_filter(y_filt, window_length=sav_window, polyorder=polyorder)

    return y_filt
