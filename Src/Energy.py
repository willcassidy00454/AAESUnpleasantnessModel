import numpy as np
from scipy.signal import savgol_filter

def getEDC(rir, sample_rate):
    integration_limit_samples = len(rir)
    reversed_rir = rir[::-1]

    # Calculate Schroeder decay
    edc_dB_reversed = 10.0 * np.log10(np.cumsum(np.square(reversed_rir)) / np.sum(np.square(rir)))
    edc_dB = edc_dB_reversed[::-1]

    time_values_samples = range(integration_limit_samples)
    time_values_seconds = [time_value / sample_rate for time_value in time_values_samples]

    return edc_dB, time_values_seconds


def getEnergyTimeCurve(rir, sample_rate, window_duration_ms: float = 10.0):
    rir /= np.max(np.abs(rir))
    window_length_samples = int((sample_rate * window_duration_ms) / 1000)
    num_rir_samples = len(rir)
    energy_time_curve = np.zeros(int(num_rir_samples / window_length_samples))
    squared_rir = np.square(rir)

    for window_index, sample_index in enumerate(range(0, int(num_rir_samples - window_length_samples), window_length_samples)):
        # # # apply windowing function here
        mean = np.mean(squared_rir[sample_index:sample_index + window_length_samples])
        energy_time_curve[window_index] = 10 * np.log10(mean)

    time_values = [(energy_bin * window_length_samples) / sample_rate for energy_bin in range(len(energy_time_curve))]

    return energy_time_curve, time_values


def getEnergySpectrum(rir, sample_rate, fft_size, etc_window_duration_ms):
        # Compute energy time curve and divide by the smoothed version
        etc, etc_times = getEnergyTimeCurve(rir, sample_rate, etc_window_duration_ms)
        smoothed_etc = savgol_filter(etc, window_length=100, polyorder=2)
        etc_over_smoothed = etc / smoothed_etc

        # Subtract mean
        etc_over_smoothed_sub_mean = etc_over_smoothed - np.mean(etc_over_smoothed)

        # Get magnitude of energy spectrum
        energy_spectrum = np.fft.rfft(etc_over_smoothed_sub_mean, n=fft_size)

        return abs(energy_spectrum)