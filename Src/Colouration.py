import numpy as np
import matplotlib.pyplot as plt
import RT
import Utils
import Energy


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

    # Normalise RIR
    rir /= np.max(np.abs(rir))

    # Estimate RT from -15 dB to -35 dB for each octave band,
    # ensuring -35 dB occurs at least 10 dB above noise floor # # # (not currently done)
    rir_bands, rir_band_centres = Utils.getOctaveBandsFromIR(rir, sample_rate)
    compensated_band_rirs = np.zeros_like(rir_bands)
    band_rts = np.zeros(len(rir_band_centres))
    for band_index in range(rir_bands.shape[1]):
        band_rir = rir_bands[:, band_index]
        band_rts[band_index] = RT.estimateRT(band_rir, sample_rate, -15, -30)

        # Window the RIR between the -15 dB and -35 dB times (assert the start should be after the mixing time)
        edc_dB = Energy.getEDC(band_rir, sample_rate)
        minus_15_position_samples = Utils.findIndexOfClosest(edc_dB, -15)
        minus_35_position_samples = Utils.findIndexOfClosest(edc_dB, -30)

        rir_sample_indices = range(rir_num_samples)
        rir_sample_indices_windowed = rir_sample_indices[minus_15_position_samples:minus_35_position_samples]

        # Compensate for IR decay shape (multiply IR by exp(6.91 * t / RT))
        sampling_period = 1.0 / sample_rate
        band_rir_windowed_compensated = [band_rir[sample_index] * np.exp(6.91 * sample_index * sampling_period / band_rts[band_index])
                                 for sample_index in rir_sample_indices_windowed]

        compensated_band_rirs[:len(band_rir_windowed_compensated), band_index] = np.divide(band_rir_windowed_compensated, np.max(np.abs(band_rir_windowed_compensated)))

    compensated_rir = np.trim_zeros(np.sum(compensated_band_rirs, axis=1))

    mean_rt = np.mean(band_rts)

    # Get magnitude spectrum
    mag_spectrum = np.abs(np.fft.rfft(compensated_rir))

    # Truncate result (Schroeder frequency lower, 4 kHz upper) and convert spectrum to log frequency
    room_volume = 5000 # this should be estimated somehow
    schroeder_frequency = 2000.0 * np.sqrt(mean_rt / room_volume)
    upper_frequency_limit = 4000

    mag_spectrum_log_trunc, mag_spectrum_freqs = Utils.linearToLog(mag_spectrum, sample_rate, schroeder_frequency, upper_frequency_limit)

    # Get smoothed spectrum
    num_octaves = np.log10(mag_spectrum_freqs[-1] / mag_spectrum_freqs[0]) / np.log10(2)
    window_size = int((len(mag_spectrum_log_trunc) / num_octaves) * 0.15) # Take 0.15 octave bands
    window = np.hamming(window_size)
    window /= np.sum(window ** 2)

    mirrored_bins_start = mag_spectrum_log_trunc[window_size:0:-1]
    mirrored_bins_end = mag_spectrum_log_trunc[:-window_size-1:-1]

    mag_spectrum_to_smooth = np.concat([mirrored_bins_start, mag_spectrum_log_trunc, mirrored_bins_end])
    mag_spectrum_smoothed = np.convolve(mag_spectrum_to_smooth, window, 'same')
    mag_spectrum_smoothed = mag_spectrum_smoothed[window_size:-window_size]

    # Divide magnitude spectrum by smoothed
    mag_over_means = mag_spectrum_log_trunc / mag_spectrum_smoothed

    # Output standard deviation of result
    colouration_score = np.std(mag_over_means)

    if should_show_plots:
        showPlots(rir,
                  colouration_score,
                  mag_spectrum_log_trunc,
                  mag_spectrum_smoothed,
                  mag_over_means,
                  mag_spectrum_freqs)

    return colouration_score
