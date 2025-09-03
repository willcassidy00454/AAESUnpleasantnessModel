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
    plt.xlabel(f"Labelled {feature} (0-9)")
    plt.ylabel(f"{feature} Feature Score")
    plt.title(f"{feature} (R-squared = {round(r_value ** 2, 2)})")
    plt.show()


def predictUnpleasantnessFromRIR(rir_filepath):
    sample_rate, spatial_rir = wavfile.read(rir_filepath)
    # spatial_rir = np.float32(spatial_rir) # Can't remember why I included this...

    # Compute features
    colouration_score = Colouration.getColouration(spatial_rir[:, 0], sample_rate, False)
    asymmetry_score = SDM.getSpatialAsymmetryScore(spatial_rir, sample_rate, False)
    flutter_echo_score = FlutterEcho.getFlutterEchoScore(spatial_rir[:, 0], sample_rate, False)

    return predictUnpleasantnessFromFeatures(colouration_score, asymmetry_score, flutter_echo_score)


def predictUnpleasantnessFromFeatures(colouration_score, asymmetry_score, flutter_echo_score, prog_item):
    if prog_item == 1:
        y_intercept = 100.037
        colouration_gradient = -43.811
        flutter_gradient = 1.225
        asymmetry_gradient = -16.976
    elif prog_item == 2:
        y_intercept = 102.412
        colouration_gradient = -72.754
        flutter_gradient = 8.161
        asymmetry_gradient = 14.831
    else:
        assert False

    linear_model = (y_intercept
                    + colouration_gradient * colouration_score
                    + asymmetry_gradient * asymmetry_score
                    + flutter_gradient * flutter_echo_score)

    return linear_model


def evaluateLinearModelOnMedians():
    x = 0


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
    # predictUnpleasantnessFromRIR(f"/Users/willcassidy/Development/GitHub/AAUnpleasantnessModel/Audio/{filename}")

    # sample_rate, spatial_rir = wavfile.read("/Users/willcassidy/Development/GitHub/AAUnpleasantnessModel/Audio/Labelled Flutter Echo/8.wav")
    # print(FlutterEcho.getFlutterEchoScore(spatial_rir[:, 0], sample_rate, True))

    evaluateFeature("Colouration")
    evaluateFeature("Spatial Asymmetry")
    evaluateFeature("Flutter Echo")

