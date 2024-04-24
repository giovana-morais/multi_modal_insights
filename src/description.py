import transformers

from PIL import Image

import config

class DatasetDescription:
    def __init__(self, descriptions, tokenizer, model, model_id, verbose=False):
        self.descriptions = descriptions
        self.tokenizer = tokenizer(model_id)
        self.model = model(model_id)
        self.verbose = verbose
        return

    def generate_input_text(self, verbose=False):
        dataset_input_text = f"The following items are descriptions of {len(self.descriptions)} items of the same dataset. Create a unique description of the whole dataset for me that summarizes the common characteristics."

        for idx, desc in enumerate(self.descriptions):
            d = f"\n{idx+1})  {desc}"
            dataset_input_text += d

        dataset_input_text += "\n"

        if self.verbose:
            print(dataset_input_text)
        return dataset_input_text

    def generate_description(self):
        dataset_input_text = self.generate_input_text()

        input_ids = self.tokenizer(dataset_input_text, return_tensors="pt").input_ids

        output = self.model.generate(input_ids, max_length=200).squeeze()
        dataset_description = self.tokenizer.decode(output)

        if self.verbose:
            print(dataset_description)
        return dataset_description


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



if __name__ == "__main__":
    import glob
    import random

    # dataset sample descriptions
    all_samples = glob.glob("/home/gigibs/Documents/Deep_Learning_Final_BS3/data/dataset/val/video_01000/*.png")

    samples = random.choices(all_samples, k=10)

    descriptions = ImageDescription(
            samples=samples,
            processor=config.IMAGE_PROCESSOR,
            model=config.IMAGE_MODEL,
            model_id=config.IMAGE_MODEL_ID,
            verbose=True
    ).generate_sample_descriptions(config.IMAGE_SAMPLE_DESCRIPTION_TEXT)

    # combining sample descriptions to generate dataset description
    dds = DatasetDescription(
            descriptions=descriptions,
            tokenizer=config.DATASET_TOKENIZER,
            model=config.DATASET_MODEL,
            model_id=config.DATASET_MODEL_ID,
            verbose=True)

    dds.generate_description()
