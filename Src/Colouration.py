import numpy as np
import statistics
import matplotlib.pyplot as plt
import RT
import Utils
import Energy
from scipy.signal import savgol_filter


def showPlots(colouration_score, mag_spectrum_log_trunc, mag_spectrum_smoothed, mag_over_means, mag_spectrum_freqs):
    plt.figure()
    fig, axes = plt.subplots(2)
    fig.set_layout_engine("tight")
    plt.suptitle(f"Colouration = {round(colouration_score, 3)}")
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

    # Normalise RIR
    rir /= np.max(np.abs(rir))

    # Estimate RT from -15 dB to -35 dB, ensuring -35 dB occurs at least 10 dB above noise floor
    rt = RT.estimateRT(rir, sample_rate, -15, -35)

    # Window the RIR between the -15 dB and -35 dB times (assert the start should be after the mixing time)
    edc_dB = Energy.getEDC(rir, sample_rate)
    minus_15_position_samples = Utils.findIndexOfClosest(edc_dB, -15)
    minus_35_position_samples = Utils.findIndexOfClosest(edc_dB, -35)

    rir_sample_indices = range(rir_num_samples)
    rir_sample_indices_windowed = rir_sample_indices[minus_15_position_samples:minus_35_position_samples]

    # Compensate for IR decay shape (multiply IR by exp(6.91 * t / RT))
    sampling_period = 1.0 / sample_rate
    rir_windowed_compensated = [rir[sample_index] * np.exp(6.91 * sample_index * sampling_period / rt)
                             for sample_index in rir_sample_indices_windowed]

    # # # # # A non-rectangular windowing function needs applying at the IR extremes to reduce frequency artefacts

    # Get magnitude spectrum
    mag_spectrum_mirrored = abs(np.fft.fft(rir_windowed_compensated))
    mag_spectrum = mag_spectrum_mirrored[0:(len(rir_windowed_compensated) // 2) - 1]

    # Truncate result (Schroeder frequency lower, 4 kHz upper) and convert spectrum to log frequency
    room_volume = 5000 # this should be estimated somehow
    schroeder_frequency = 2000.0 * np.sqrt(rt / room_volume)
    upper_frequency_limit = 4000

    mag_spectrum_log_trunc, mag_spectrum_freqs = Utils.linearToLog(mag_spectrum, sample_rate, schroeder_frequency, upper_frequency_limit)

    # Get smoothed spectrum
    window_length_bins = len(mag_spectrum_log_trunc) // 20
    mag_spectrum_smoothed = savgol_filter(mag_spectrum_log_trunc, window_length_bins, 1)

    # Divide magnitude spectrum by smoothed
    mag_over_means = mag_spectrum_log_trunc / mag_spectrum_smoothed

    # Output standard deviation of result
    colouration_score = statistics.stdev(mag_over_means)

    if should_show_plots:
        showPlots(colouration_score,
                  mag_spectrum_log_trunc,
                  mag_spectrum_smoothed,
                  mag_over_means,
                  mag_spectrum_freqs)

    return colouration_score
