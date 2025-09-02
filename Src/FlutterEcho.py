import Utils
import numpy as np
from matplotlib import pyplot as plt
import Energy
from scipy.signal import butter, sosfilt


def showEnergySpectrumPlots(num_octave_bands, energy_spectra, energy_spectrum_freqs, octave_band_centres, flutter_score):
    fig, axes = plt.subplots(num_octave_bands)
    fig.set_size_inches(6, 8)
    fig.set_layout_engine("tight")
    plt.suptitle(f"|FFT(Energy Decay Fluctuations)| (flutter = {round(flutter_score, 3)})")

    for octave_band in range(num_octave_bands):
        energy_spectrum = energy_spectra[:, octave_band]
        axes[octave_band].plot(energy_spectrum_freqs, energy_spectrum)
        axes[octave_band].set_title(f"{octave_band_centres[octave_band]} Hz")

    plt.show()

def showACFPlots(num_octave_bands, auto_correlations, sample_rate, octave_band_centres, flutter_score, etc_window_duration_ms):
    fig, axes = plt.subplots(num_octave_bands)
    fig.set_size_inches(6, 8)
    fig.set_layout_engine("tight")
    plt.suptitle(f"Auto-Correlation Function of Energy Decay (flutter = {round(flutter_score, 3)})")

    times = np.arange(0, auto_correlations.shape[0]) * (etc_window_duration_ms / 1000)
    # frequencies = 1 / (np.clip(bin_indices, 0.00001, None) )

    for octave_band in range(num_octave_bands):
        auto_correlation = auto_correlations[:, octave_band]
        # axes[octave_band].plot(times, 20 * np.log10(np.clip(auto_correlation, 0.00001, 1)))
        axes[octave_band].plot(times, auto_correlation)
        axes[octave_band].set_title(f"{octave_band_centres[octave_band]} Hz")
        axes[octave_band].set_xlim([0.05, 0.5])
        # axes[octave_band].set_ylim([-200, 400])

    plt.show()

def getFlutterEchoScore(rir, sample_rate, should_show_plots=False):
    # Truncate start of RIR up to max magnitude
    rir = rir[np.argmax(abs(rir)):]

    # High-pass RIR from 6 kHz
    filter_order = 4
    cutoff_Hz = 7000
    sos = butter(2 * filter_order, cutoff_Hz, 'highpass', fs=sample_rate, output='sos')
    rir_high_passed = sosfilt(sos, rir)

    # Get energy spectrum of each octave band, truncate between 2 and 20 Hz,
    # and calculate distance of max in each from mean, minus std dev
    fft_size = 2 ** 9
    etc_window_duration_ms = 10.0
    energy_spectrum_freqs = np.fft.rfftfreq(fft_size, etc_window_duration_ms / 1000.0)

    etc_sample_rate = 1.0 / (etc_window_duration_ms / 1000.0)

    min_flutter_frequency = 2 # Doesn't seem to be sensitive up to ~10 Hz
    max_flutter_frequency = 20

    min_energy_freq_index = int(np.floor(len(energy_spectrum_freqs) * (min_flutter_frequency / (etc_sample_rate / 2))))
    max_energy_freq_index = int(np.floor(len(energy_spectrum_freqs) * (max_flutter_frequency / (etc_sample_rate / 2))))

    # Get energy time curve of the high-passed RIR
    etc_dB, _ = Energy.getEnergyTimeCurve(rir_high_passed, sample_rate, etc_window_duration_ms)

    etc_dB[:Utils.findIndexOfClosest(etc_dB, -60)] = -60 # This might not be feasible for recorded RIRs; 40 dB might be more robust

    # Get FFT of energy time curve in decibels, truncated between 2-20 Hz
    energy_spectrum_dB = 10 * np.log10(np.abs(np.fft.rfft(etc_dB, n=fft_size)))
    energy_spectrum_dB = energy_spectrum_dB[min_energy_freq_index:max_energy_freq_index]

    # Output mean of the energy in the 2-20 Hz region (might not need to perform FFT for this)
    flutter_echo_score = np.mean(energy_spectrum_dB)

    if should_show_plots:
        showEnergySpectrumPlots(energy_spectrum_dB, energy_spectrum_freqs[min_energy_freq_index:max_energy_freq_index], octave_band_centres, flutter_echo_score)

    return flutter_echo_score