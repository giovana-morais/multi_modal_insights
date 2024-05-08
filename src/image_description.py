from bertopic.backend import MultiModalBackend
from bertopic.representation import VisualRepresentation

from .dataset_description import DatasetDescription

__EMBEDDING_MODEL__ = "clip-ViT-B-32"
__REPRESENTATION_MODEL__ = "nlpconnect/vit-gpt2-image-captioning"

class ImageDescription(DatasetDescription):
    def __init__(self, data_home=None, data=None):
        super(ImageDescription, self).__init__()
        if data_home is None and data is None:
            raise ValueError("Both `data_home` is None and `data` is None.")

        self.data_home = data_home
        self.data = data
        self.embedding_model = MultiModalBackend(__EMBEDDING_MODEL__,
                batch_size=32)
        self.representation_model = {
            "Visual_Aspect": VisualRepresentation(image_to_text_model=__REPRESENTATION_MODEL__)
        }
        self.topic_model = super().topic_model(self.embedding_model,
                self.representation_model)
        self.topic_texts = self.fit_data()
        self.prompt = self.create_prompt(self.topic_texts)
        self.dataset_description = super().dataset_description(self.prompt)

    # TODO: add the update label function
    def fit_data(self):
        print("[INFO] creating topics")
        self.topic_model.fit(documents=None, images=self.data)
        topics = self.topic_model.get_topic_info()

        # -1 are outliers, so we remove them
        # and get the top 20 topics
        topics = topics[topics["Topic"] > -1].head(20)
        topic_text = ''.join(topics.apply(lambda x: self.generate_topic_text(x.Name,
            x.Count), axis=1).to_list())

        return topic_text

    def generate_topic_text(self, name, count):
        return f"group name: {name} | count: {count} images\n"

    def create_prompt(self, topic_texts):
        system_prompt = """
            <S>[INST] <<SYS>>
            you are a helpful, respectful and honest assistant for dataset description.
            <</SYS>>
            """

        example_prompt = """
        I have a dataset that contains the following group of images:
        group name:  Cars | count: 200 images
        group name:  Children. | count: 400 images
        group name:  People runing. | count: 300 images

        Based on the information above, please create a description of this group of images. This description should be in plain text, avoiding lists and using a common vocabulary.

        [/INST] This dataset comprises a diverse collection of images featuring various subjects. Within the dataset, there are 200 images of cars, capturing a range of makes, models, and settings. Additionally, there are 400 images focused on children, potentially depicting them engaged in various activities or scenarios. Furthermore, the dataset includes 300 images showcasing people running, suggesting a theme of physical activity or sports. Together, these images offer a multifaceted view of different subjects and activities, providing ample material for analysis and exploration.
        """

        main_prompt = """
        [INST]
        I have a dataset that contains the following group of images:
        [TOPICS]

        Based on the information about the topic above, please create a short description of this group of images. This description should be in plain text, and using a common vocabulary.
        [/INST]
        """

        main_prompt = main_prompt.replace("[TOPICS]", topic_texts)

        prompt = system_prompt + example_prompt + main_prompt
        return prompt
