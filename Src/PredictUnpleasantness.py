from scipy.io import wavfile
import Colouration
import FlutterEcho


def predictUnpleasantness(rir_filepath):
    # Load audio channels and pre-process
    sample_rate, spatial_rir = wavfile.read(rir_filepath)

    # Compute metrics
    # colouration_score = Colouration.getColouration(spatial_rir[:, 0], sample_rate, True)

    flutter_echo_scores = FlutterEcho.getFlutterEchoScore(spatial_rir[:, 0], sample_rate, True)

    # Compute model
    # model_output = flutter_echo_scores

    # print(model_output)


if __name__ == "__main__":
    predictUnpleasantness("/Users/willcassidy/Development/GitHub/AAUnpleasantnessModel/Audio/Flutter.wav")