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

    flutter_echo_scores = FlutterEcho.getFlutterEchoScore(spatial_rir[:, 0], sample_rate, True)

    # sos = signal.butter(2, 40 / (sample_rate / 2), 'highpass', output='sos')
    # spatial_rir = signal.sosfilt(sos, spatial_rir)

    # SDM.plotSpatioTemporalMap(spatial_rir, sample_rate, "median", 200)
    # SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, True)

    # Compute model
    # model_output = flutter_echo_scores

    # print(model_output)


if __name__ == "__main__":
    filename = "Flutter.wav"
    predictUnpleasantness(f"/Users/willcassidy/Development/GitHub/AAUnpleasantnessModel/Audio/{filename}")