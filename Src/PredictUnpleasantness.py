from scipy.io import wavfile
import Colouration
import FlutterEcho
import numpy as np
import SDM
from scipy import signal
import matplotlib.pyplot as plt


def predictUnpleasantness(rir_filepath):
    # Load audio channels and pre-process
    sample_rate, spatial_rir = wavfile.read(rir_filepath)

    spatial_rir = np.float32(spatial_rir)

    # Compute metrics
    # colouration_score = Colouration.getColouration(spatial_rir[:, 0], sample_rate, True)

    # flutter_echo_scores = FlutterEcho.getFlutterEchoScore(spatial_rir[:, 0], sample_rate, True)

    # sos = signal.butter(2, 500 / (sample_rate / 2), 'highpass', output='sos')
    # spatial_rir = signal.sosfilt(sos, spatial_rir)

    plot_angles_rad, radii_dB = SDM.getSpatioTemporalMap(spatial_rir,
                                                         sample_rate,
                                                         start_relative_to_direct_ms=-1,
                                                         duration_ms=200,
                                                         plane="transverse",
                                                         num_plot_angles=200)

    plt.polar(plot_angles_rad, radii_dB)
    plt.show()

    # Compute model
    # model_output = flutter_echo_scores

    # print(model_output)


if __name__ == "__main__":
    predictUnpleasantness("/Users/willcassidy/Development/GitHub/AAUnpleasantnessModel/Audio/SingleLSLeft.wav")