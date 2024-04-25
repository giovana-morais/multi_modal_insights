import librosa
from gradio_client import Client

class AudioDescription:
    def __init__(self, samples, processor, model, model_id, model_kwargs=None,
            verbose=False):
        self.client = Client("https://yuangongfdu-ltu-2.hf.space/")
        self.samples = samples
        # sampling rate following model documentation
        self.sr = 16000
        return

    def generate_sample_descriptions(self, text):
        descriptions = []
        for audio_path in self.samples:
            if librosa.get_samplerate(audio_path) != self.default_sr:
                x, _ = librosa.load()

        result = client.predict(
            "path_to_your_wav/audio.wav",  # your audio file in 16K
            "",
            "What can be inferred from the audio?",    # your question
            "7B (Default)",    # str in 'LLM size' Radio component
            api_name="/predict"
        )
        print(result)
        return
