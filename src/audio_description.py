import tempfile

import librosa
import soundfile as sf
import tqdm
from gradio_client import Client

from .dataset_description import DatasetDescription

__EMBEDDING_MODEL__ = "all-MiniLM-L6-v2"

__AUDIO_DESCRIPTION_MODEL__ = "https://yuangongfdu-ltu-2.hf.space/"
__AUDIO_DESCRIPTION_PROMPT__ = "Is the audio sample music, environmental sound or speech? Can you describe it?"
__LTU_SAMPLING_RATE__ = 16000

class AudioDescription(DatasetDescription):
    def __init__(self, data_home=None, data=None, max_length=15):
        super(AudioDescription, self).__init__()
        if data_home is None and data is None:
            raise ValueError("Both `data_home` is None and `data` is None.")

        self.audio_data = data

        self.default_sr = __LTU_SAMPLING_RATE__
        self.max_length = max_length*self.default_sr

        self.audio_model = Client(__AUDIO_DESCRIPTION_MODEL__)
        self.text_data = self.generate_sample_descriptions()
        self.embedding_model = __EMBEDDING_MODEL__

        self.topic_model = super().topic_model(self.embedding_model)
        self.topic_texts = self.fit_data()
        self.prompt = self.create_prompt(self.topic_texts)
        self.dataset_description = super().dataset_description(self.prompt)

    def fit_data(self):
        print("[INFO] creating topics")
        self.topic_model.fit(self.text_data)
        topics = self.topic_model.get_topic_info()

        # -1 are outliers, so we remove them
        # and get the top 20 topics
        topics = topics[topics["Topic"] > -1].head(20)
        topic_text = ''.join(topics.apply(lambda x: self.generate_topic_text(x.Name,
            x.Count), axis=1).to_list())

        return topic_text


    def generate_topic_text(self, name, count):
        return f"group name: {name} | count: {count} audios\n"

    def create_prompt(self, topic_texts):
        system_prompt = """
            <S>[INST] <<SYS>>
            you are a helpful, respectful and honest assistant for dataset description.
            <</SYS>>
            """

        example_prompt = """
        I have a dataset that contains the following group of audio:
        group name:  Cars | count: 200 audios
        group name:  Children. | count: 400 audios
        group name:  People runing. | count: 300 audios

        Based on the information above, please create a description of this group of audio. This description should be in plain text, avoiding lists and using a common vocabulary.

        [/INST] This dataset comprises a diverse collection of audio featuring various subjects. Within the dataset, there are 200 audio of cars, capturing a range of makes, models, and settings. Additionally, there are 400 audio focused on children, potentially depicting them engaged in various activities or scenarios. Furthermore, the dataset includes 300 audio showcasing people running, suggesting a theme of physical activity or sports. Together, these audio offer a multifaceted view of different subjects and activities, providing ample material for analysis and exploration.
        """

        main_prompt = """
        [INST]
        I have a dataset that contains the following group of audio:
        [TOPICS]

        Based on the information about the topic above, please create a short description of this group of audio. This description should be in plain text, and using a common vocabulary.
        [/INST]
        """

        main_prompt = main_prompt.replace("[TOPICS]", topic_texts)

        prompt = system_prompt + example_prompt + main_prompt
        return prompt

    def crop_audio(self, x):
        cut_x = x.copy()
        cut_x = cut_x[:self.max_length]

        return cut_x

    def generate_sample_descriptions(self):
        print("[INFO] generating audio description")
        descriptions = []
        pbar = tqdm.tqdm(self.audio_data)
        pbar.set_description("Audio description")

        for audio_path in pbar:
            # tmp variable in case we need resampling
            path = audio_path
            if librosa.get_samplerate(audio_path) != self.default_sr:
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
                    __AUDIO_DESCRIPTION_PROMPT__,
                    "7B (Default)",
                    api_name="/predict"
                )

            descriptions.append(description)

        return descriptions
