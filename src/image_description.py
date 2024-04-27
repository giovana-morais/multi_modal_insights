import transformers
from PIL import Image

class ImageDescription:
    def __init__(self, samples, processor, model, model_id, model_kwargs=None, verbose=False):
        """
        arguments
        ---
            samples         : list[str]
                list with samples paths
            processor       : transformers.processor
            model           : transformers.model
            model_kwargs    : dict
                dictionary with additional model parameters
            model_id        : str
                identifier of model
        """

        self.samples = samples
        if len(samples) < 10:
            raise ValueError("We need at least 10 samples")


        self.processor = processor(model_id)
        self.model = model(model_id)

        return


    def generate_sample_descriptions(self, text):
        """
        generate description for every sample image

        arguments
        ---
            text : str
                base text for processor

        return
        ---
            descriptions : list[str]
                list with description for each sample
        """

        descriptions = []
        for img_path in self.samples:
            image = Image.open(img_path).convert("RGB")
            input = self.processor(image, text, return_tensors="pt")
            model_output = self.model.generate(**input, max_length=40).squeeze()
            description = self.processor.decode(model_output, skip_special_tokens=True)
            descriptions.append(description)

        return descriptions


    def metadata(self, data_home):
        return
