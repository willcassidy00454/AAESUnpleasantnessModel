from scipy.io import wavfile
import Colouration
import FlutterEcho
import numpy as np
from scipy import stats
import SDM
import SpectralEvolution
from scipy import signal
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile


# Reads the RIR files in folder "Labelled {feature}", the names of which are ranked from 0-10
# (e.g. "0.wav", "0_1.wav", "1.wav"), and compares these to the feature outputs for the RIRs.
# feature = "Colouration" | "Spatial Asymmetry" | "Flutter Echo"
def evaluateFeature(feature="Colouration"):
    labelled_examples_dir = f"/Users/willcassidy/Development/GitHub/AAUnpleasantnessModel/Audio/Labelled {feature}/"
    filenames = [filename for filename in sorted(listdir(labelled_examples_dir)) if isfile(labelled_examples_dir + filename) and filename.endswith("wav")]

    labels = [np.floor(float(label.strip(".wav").replace("_", "."))) for label in filenames]
    feature_outputs = np.zeros_like(filenames)

    for file_index, filename in enumerate(filenames):
        filepath = labelled_examples_dir + filename
        sample_rate, spatial_rir = wavfile.read(filepath)

        if feature == "Colouration":
            feature_outputs[file_index] = Colouration.getColouration(spatial_rir[:, 0], sample_rate, False)
        elif feature == "Spatial Asymmetry":
            feature_outputs[file_index] = SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, False)
        elif feature == "Flutter Echo":
            feature_outputs[file_index] = FlutterEcho.getFlutterEchoScore(spatial_rir[:, 0], sample_rate, False)
        else:
            assert False

    feature_outputs = [float(output) for output in feature_outputs]

    gradient, y_intercept, r_value, p_value, std_err = stats.linregress(labels, feature_outputs)
    linear_regression = np.poly1d([gradient, y_intercept])

    plt.plot(labels, feature_outputs, 'o', labels, linear_regression(labels))
    plt.xlabel(f"Labelled {feature} (0-10)")
    plt.ylabel(f"{feature} Feature Score")
    plt.title(f"{feature} (R-squared = {round(r_value ** 2, 2)})")
    plt.show()

def predictUnpleasantness(rir_filepath):
    # Load audio channels and pre-process
    sample_rate, spatial_rir = wavfile.read(rir_filepath)

    spatial_rir = np.float32(spatial_rir)

    # Compute features
    # colouration_score = Colouration.getColouration(spatial_rir, sample_rate, True)

    spectral_score = SpectralEvolution.getSpectralEvolutionScore(spatial_rir[:, 0], sample_rate, True)

    print(spectral_score)

    # flutter_echo_score = FlutterEcho.getFlutterEchoScore(spatial_rir[:, 0], sample_rate, True)

    # sos = signal.butter(2, 40 / (sample_rate / 2), 'highpass', output='sos')
    # spatial_rir = signal.sosfilt(sos, spatial_rir)

    # SDM.plotSpatioTemporalMap(spatial_rir, sample_rate, "median", 50)
    # asymmetry_score = SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, True)

    # Compute model
    # model_output = flutter_echo_scores

    # print(model_output)


if __name__ == "__main__":
    # filename = "Flutter.wav" # high flutter
    # filename = "Room3.wav" # pretty high flutter, front-back though
    # filename = "PassiveRoom.wav" # fairly high flutter, late
    # filename = "horizontal_1.wav" # near zero flutter
    # filename = "NoiseIR_RT_1s_48k_1.wav" # zero flutter
    # filename = "Room33.wav"
    # filename = "SingleLSLeft.wav"
    # filename = "Wet.wav"
    # filename = "BrightLate.wav"
    # filename = "BrightLateColoured.wav"
    # filename = "DullLate.wav"

    # Passive Rooms
    # filename = "Passive11.wav"
    # filename = "Room3.wav"
    # filename = "PassiveRoom.wav"
    # filename = "Pilsen.wav"
    # filename = "Pilsen2.wav"
    # filename = "Pilsen3.wav"
    # filename = "Normal.wav"
    # predictUnpleasantness(f"/Users/willcassidy/Development/GitHub/AAUnpleasantnessModel/Audio/{filename}")

    evaluateFeature("Colouration")