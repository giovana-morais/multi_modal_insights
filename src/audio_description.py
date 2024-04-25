import librosa
import soundfile as sf
from gradio_client import Client

class AudioDescription:
    def __init__(self, samples):
        self.client = Client("https://yuangongfdu-ltu-2.hf.space/")
        self.samples = samples
        # sampling rate following model documentation
        self.default_sr = 16000
        self.max_length = 15*self.default_sr
        return

    def generate_sample_descriptions(self, text):
        descriptions = []
        for audio_path in self.samples:
            # tmp variable in case we need resampling
            path = audio_path
            if librosa.get_samplerate(audio_path) != self.default_sr:
                print("resampling and cropping audio")
                # load and resample audio
                x, _ = librosa.load(audio_path, sr=self.default_sr)
                # if audio is too big, crop 15 seconds
                if len(x) > self.max_length:
                    x = x[:self.max_length]
                sf.write("tmp.wav", x, self.default_sr)
                path = "tmp.wav"

            description = self.client.predict(
                path,  # your audio file in 16K
                "",
                text,    # your question
                "7B (Default)",    # str in 'LLM size' Radio component
                api_name="/predict"
            )
            descriptions.append(description)

        return descriptions
