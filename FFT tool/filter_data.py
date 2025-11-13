import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, medfilt, detrend, savgol_filter, sosfreqz, freqz

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
                      do_bp=False, bp_low=10.0, bp_high=100.0, bp_order=4,
                      do_savgol=False, sav_window=11, polyorder=3,
                      return_intermediates=False):
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
        median_k = medfilt_kernel_safe(median_k, len(y_filt))
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

    # 5b️⃣ Bandpass filter (alternative to HP+LP combination)
    if do_bp:
        sos_bp = butter_sos(lowcut=bp_low, highcut=bp_high, fs=fs, order=bp_order)
        y_filt = apply_sos_filter(y_filt, sos_bp)

    # 6️⃣ Optional Savitzky–Golay smoothing
    if do_savgol:
        sav_window, polyorder = savgol_params_safe(sav_window, polyorder, len(y_filt))
        y_filt = savgol_filter(y_filt, window_length=sav_window, polyorder=polyorder)

    if return_intermediates:
        return y_filt, fs
    return y_filt
def medfilt_kernel_safe(k, n):
    k = int(k)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    k = min(k, n)
    if k % 2 == 0:
        k = max(1, k - 1)
    if k < 1:
        k = 1
    return k


def savgol_params_safe(window, polyorder, n):
    window = int(window)
    polyorder = int(polyorder)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    window = min(window, n)
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 1:
        window = 3
    polyorder = max(1, polyorder)
    if polyorder >= window:
        polyorder = window - 1
    if polyorder < 1:
        polyorder = 1
    return window, polyorder


def compute_combined_freq_response(freqs,
                                   fs,
                                   do_hp=True, hp_cut=0.1, hp_order=3,
                                   do_notch=False, notch_freqs=None, notch_Q=30,
                                   do_lp=True, lp_cut=250.0, lp_order=4,
                                   do_bp=False, bp_low=10.0, bp_high=100.0, bp_order=4):
    """
    Compute the combined frequency response (complex) of the linear filters
    that are enabled in the preprocessing pipeline. Non-linear filters such as
    median and Savitzky–Golay are ignored because a classical frequency response
    does not exist or is data dependent.

    Args:
        freqs: Array of frequency samples (Hz or 1/X units) at which to evaluate.
        fs: Sampling rate corresponding to the data (same units as freq axis).
        Remaining parameters mirror the filter enable flags and parameters in
        preprocess_signal.

    Returns:
        complex ndarray of the same shape as freqs representing the combined
        frequency response. Returns None if freqs is empty.
    """
    if freqs is None or len(freqs) == 0 or fs <= 0:
        return None

    # Convert target frequencies to digital rad/sample
    w = 2.0 * np.pi * np.asarray(freqs) / fs
    # Clip to the valid digital frequency range [0, π]
    w = np.clip(w, 0.0, np.pi - 1e-9)

    response = np.ones_like(w, dtype=np.complex128)

    if do_hp:
        sos_hp = butter_sos(lowcut=hp_cut, highcut=None, fs=fs, order=hp_order)
        _, h_hp = sosfreqz(sos_hp, worN=w)
        response *= h_hp

    if do_notch and notch_freqs:
        for f0 in notch_freqs:
            b, a = iirnotch(f0 / (0.5 * fs), notch_Q)
            _, h_notch = freqz(b, a, worN=w)
            response *= h_notch

    if do_lp:
        sos_lp = butter_sos(lowcut=None, highcut=lp_cut, fs=fs, order=lp_order)
        _, h_lp = sosfreqz(sos_lp, worN=w)
        response *= h_lp

    if do_bp:
        sos_bp = butter_sos(lowcut=bp_low, highcut=bp_high, fs=fs, order=bp_order)
        _, h_bp = sosfreqz(sos_bp, worN=w)
        response *= h_bp

    return response
