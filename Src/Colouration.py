import numpy as np
import matplotlib.pyplot as plt
import RT
import Utils
import Energy
from scipy.signal import savgol_filter


def showPlots(rir, colouration_score, mag_spectrum_log_trunc, mag_spectrum_smoothed, mag_over_means, mag_spectrum_freqs):
    plt.figure()
    fig, axes = plt.subplots(2)
    fig.set_layout_engine("tight")
    plt.suptitle(f"Colouration (stddev of bottom plot) = {round(colouration_score, 3)}")
    axes[0].set_xscale("log")
    axes[0].set_title('Magnitude Spectrum Raw (dashed) and Smoothed (solid)')
    axes[0].plot(mag_spectrum_freqs, mag_spectrum_log_trunc, 'c--')
    axes[0].plot(mag_spectrum_freqs, mag_spectrum_smoothed, 'black')
    axes[0].set_xticks([20, 200, 2000])
    axes[0].set_xticklabels(["20", "200", "2k"])
    axes[1].set_xscale("log")
    axes[1].set_title('Magnitude Spectrum Raw/Smoothed')
    axes[1].plot(mag_spectrum_freqs, mag_over_means, 'black')
    axes[1].set_xticks([20, 200, 2000])
    axes[1].set_xticklabels(["20", "200", "2k"])
    plt.show()


def getColouration(rir, sample_rate, should_show_plots=False):
    rir_num_samples = len(rir)

    # Estimate RT from -5 dB to -40 dB ensuring -40 dB occurs at least 10 dB above noise floor
    # # # (not currently done) (modification; previously -15 to -35 dB)
    rt = RT.estimateRT(rir, sample_rate, start_dB=-5, end_dB=-40)

    # Window the RIR between the 0 dB and -40 dB times (modification; assert the start should be after the mixing time,
    # previously -15 to -35 dB)
    edc_dB, time_values = Energy.getEDC(rir, sample_rate)
    minus_15_position_samples = Utils.findIndexOfClosest(edc_dB, 0)
    minus_35_position_samples = Utils.findIndexOfClosest(edc_dB, -40)

    rir_sample_indices = range(rir_num_samples)
    rir_sample_indices_windowed = rir_sample_indices[minus_15_position_samples:minus_35_position_samples]

    # Compensate for IR decay shape (multiply IR by exp(6.91 * t / RT))
    sampling_period = 1.0 / sample_rate
    rir_windowed_compensated = [rir[sample_index] * np.exp(6.91 * sample_index * sampling_period / rt)
                                for sample_index in rir_sample_indices_windowed]

    # Get magnitude spectrum
    fft_size = 2 ** 16
    mag_spectrum = np.abs(np.fft.rfft(rir_windowed_compensated, fft_size))

    # Truncate result (Schroeder frequency lower, 2 kHz upper) and convert spectrum to log frequency
    room_volume = 5000 # assumed
    schroeder_frequency = 2000.0 * np.sqrt(rt / room_volume)
    upper_frequency_limit = 2000 # modified from 4 kHz

    mag_spectrum_log_trunc, mag_spectrum_freqs = Utils.linearToLog(mag_spectrum, sample_rate, schroeder_frequency, upper_frequency_limit)

    # Convert magnitude to decibels (modification)
    mag_spectrum_log_trunc = 10 * np.log10(mag_spectrum_log_trunc)

    # Get smoothed spectrum, mirroring start and ends for one window length to avoid edge effects
    num_octaves = np.log10(mag_spectrum_freqs[-1] / mag_spectrum_freqs[0]) / np.log10(2)
    window_size = int((len(mag_spectrum_log_trunc) / num_octaves) * 0.15) # Smooth in 0.15 * octave bands
    mirrored_bins_start = mag_spectrum_log_trunc[window_size:0:-1]
    mirrored_bins_end = mag_spectrum_log_trunc[:-window_size-1:-1]

    mag_spectrum_to_smooth = np.concat([mirrored_bins_start, mag_spectrum_log_trunc, mirrored_bins_end])
    mag_spectrum_smoothed = savgol_filter(mag_spectrum_to_smooth, window_size, 1)
    mag_spectrum_smoothed = mag_spectrum_smoothed[window_size:-window_size]

    # Subtract smoothed magnitude from raw (modification; use divide for standard)
    mag_minus_mean = mag_spectrum_log_trunc - mag_spectrum_smoothed

    # Clip below 0 to remove notch effects due to dB scale (modification)
    mag_minus_mean = np.clip(mag_minus_mean, 0, None)

    # Output summation of standard deviation and peakedness (modification)
    std_dev = np.std(mag_minus_mean)
    peakedness = 10 * np.log10(np.max(mag_minus_mean) - np.mean(mag_minus_mean) - std_dev)
    colouration_score = std_dev + peakedness / 10

    # Scale to approximately 0-1 (modification)
    colouration_score = (colouration_score - 1.5) / 0.9

    if should_show_plots:
        showPlots(rir,
                  colouration_score,
                  mag_spectrum_log_trunc,
                  mag_spectrum_smoothed,
                  mag_minus_mean,
                  mag_spectrum_freqs)

    return colouration_score
