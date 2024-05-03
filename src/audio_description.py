import tempfile

import librosa
import soundfile as sf
from gradio_client import Client

from .dataset_description import DatasetDescription

__AUDIO_MODEL__ = "https://yuangongfdu-ltu-2.hf.space/"
__AUDIO_PROMPT__ = "Is the audio sample music, environmental sound or speech? Can you describe it?"
__LTU_SAMPLING_RATE__ = 16000

class AudioDescription(DatasetDescription):
    def __init__(self, data_home, samples, max_length=15):
        super(AudioDescription, self).__init__()
        self.data_home = data_home
        self.audio_model = Client(__AUDIO_MODEL__)
        self.samples = samples
        # sampling rate following model documentation
        self.sr = __LTU_SAMPLING_RATE__
        self.max_length = max_length*self.default_sr
        self.descriptions = self.generate_sample_descriptions()
        self.dataset_description = super().dataset_description(self.descriptions)
        return

    def crop_audio(self, x):
        cut_x = x.copy()
        cut_x = cut_x[:self.max_length]

        return cut_x

    def generate_sample_descriptions(self):
        descriptions = []
        for audio_path in self.samples:
            # tmp variable in case we need resampling
            path = audio_path
            if librosa.get_samplerate(audio_path) != self.default_sr:
                print("resampling and cropping audio")
                # load and resample audio
                x, _ = librosa.load(audio_path, sr=self.default_sr)
                # if audio is too big, crop 15 seconds
                x = self.crop_audio(x)

            # generate tmp file for the cropped audio and send it to model
            # file is destroyed afterwards
            with tempfile.NamedTemporaryFile(dir=".", suffix=".wav") as f:
                sf.write(f.name, x, self.default_sr)
                description = self.audio_model.predict(
                    f.name,
                    "",
                    __AUDIO_PROMPT__,
                    "7B (Default)",
                    api_name="/predict"
                )

            descriptions.append(description)

        return descriptions
