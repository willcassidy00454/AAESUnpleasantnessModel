import numpy as np
from scipy.interpolate import interp1d


def findIndexOfClosest(list, target):
    list = np.asarray(list)
    index_of_closest = (np.abs(list - target)).argmin()
    return index_of_closest


def truncateSpectrum(spectrum, sample_rate, min_frequency, max_frequency):
    num_bins = len(spectrum)
    nyquist_frequency = sample_rate / 2.0
    bin_width = nyquist_frequency / num_bins
    min_frequency_index = int(np.floor(min_frequency / bin_width))
    max_frequency_index = int(np.floor(max_frequency / bin_width))
    truncated_spectrum = spectrum[min_frequency_index:max_frequency_index]
    return truncated_spectrum


def interpolateList(list_to_interpolate, new_length):
    delta = (len(list_to_interpolate) - 1) / (new_length - 1)
    interpolated_list = np.zeros(new_length)

    for new_position in range(new_length):
        index_integer, index_fraction = int(new_position * delta // 1), new_position * delta % 1  # Split floating-point index into whole & fractional parts
        end_index = index_integer + 1 if index_fraction > 0 else index_integer  # Avoid index error
        interpolated_list[new_position] = ((1 - index_fraction) * list_to_interpolate[index_integer]
                                           + index_fraction * list_to_interpolate[end_index])

    return interpolated_list


def linearToLog(magnitudes, sample_rate, f_min, f_max):
    """
    Convert linear FFT magnitudes to logarithmic frequency space.

    Parameters:
        magnitudes (array): FFT magnitudes (assumes only positive frequencies from 0 Hz to Nyquist).
        sample_rate (float): Sampling rate in Hz.
        num_bins (int): Number of logarithmic frequency bins.
        f_min (float): Minimum frequency for log space.
        f_max (float): Maximum frequency, default is Nyquist.

    Returns:
        log_mags (array): Interpolated magnitudes in log frequency space between f_min and f_max
    """
    num_bins = len(magnitudes)
    nyquist = sample_rate / 2
    if f_max is None:
        f_max = nyquist

    # Linear frequency axis (positive frequencies only)
    lin_freqs = np.linspace(0, nyquist, num_bins)

    # Remove DC component (0 Hz) to avoid log(0)
    lin_freqs = lin_freqs[1:]
    magnitudes = magnitudes[1:]

    # Logarithmic frequency axis
    log_freqs = np.logspace(np.log10(f_min), np.log10(f_max), num_bins)

    # Interpolation
    interp = interp1d(lin_freqs, magnitudes, kind='linear', bounds_error=False, fill_value=0.0)
    log_mags = interp(log_freqs)

    return log_mags, log_freqs


def getMidpointsBetween(values):
    return [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]


def getThirdOctaveBandCentres(min_frequency, max_frequency):
    reference_frequency = 1000.0
    increase_factor = 2 ** (1 / 3)

    # Find lower and upper band indices relative to 1000 Hz
    min_band_index = int(np.ceil(np.log10(min_frequency / reference_frequency) / np.log10(increase_factor)))
    max_band_index = int(np.floor(np.log10(max_frequency / reference_frequency) / np.log10(increase_factor)))

    # Calculate centre frequencies
    centre_frequencies = [reference_frequency * (increase_factor ** n) for n in range(min_band_index, max_band_index + 1)]

    return centre_frequencies


def getMeanMagnitudeBetweenFrequencies(half_spectrum_magnitudes, sample_rate, min_frequency, max_frequency):
    magnitudes_between = truncateSpectrum(half_spectrum_magnitudes, sample_rate, min_frequency, max_frequency)
    mean = np.mean(magnitudes_between)
    return mean