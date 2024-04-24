import transformers
import torchvision.datasets as dset

from PIL import Image
# from torch.utils.data import Dataset, DataLoader

class Description:
    def __init__(self, samples, model_id):
        self.data_home = data_home
        self.data_type = data_type
        return


    def generate_description():
        return

class ImageDescription:
    def __init__(self, samples, model_id, model_kwargs=None):
        # super(ImageDescription, self).__init__()
        self.samples = samples
        if len(samples) < 10:
            raise ValueError("We need at least 10 samples")


        self.model_id = model_id
        self.model_kwargs = model_kwargs

        self.pipeline = transformers.pipeline(
                "image-to-text",
                model=model_id,
                model_kwargs=model_kwargs
        )


        return

    def generate_sample_descriptions(self, text, processor, model):
        print("instatiating modality-specific model")

        processor = processor(self.model_id)
        model = model(self.model_id)

        descriptions = []
        for img_path in self.samples:
            image = Image.open(img_path).convert("RGB")
            input = processor(image, text, return_tensors="pt")
            model_output = model.generate(**input, max_length=40).squeeze()
            # print(model_output.shape)
            # print(model_output.squeeze().shape)
            description = processor.decode(model_output, skip_special_tokens=True)
            descriptions.append(description)

        return descriptions

    # def generate_dataset_description():
    #     return


# class AudioDescription():
#     def __init__(self, dataloader, n_samples, model_id, model_kwargs=None):
#         self.n_samples = n_samples
#         self.model_id = model_id
#         self.model_kwargs = model_kwargs

#         self.pipeline = transformers.pipeline(
#                 "image-to-text",
#                 model=model_id,
#                 model_kwargs=model_kwargs
#         }
#         return

def generate_input_text(descriptions):
    dataset_input_text = f"The following items are descriptions of {len(descriptions)} items of the same dataset. Create a unique description of the whole dataset for me that summarizes the common characteristics."

    for idx, desc in enumerate(descriptions):
        d = f"\n{idx+1})  {desc}"
        dataset_input_text += d

    return dataset_input_text


if __name__ == "__main__":
    import glob
    import random

    # dataset sample descriptions
    all_samples = glob.glob("/home/gigibs/Documents/Deep_Learning_Final_BS3/data/dataset/val/video_01000/*.png")

    samples = random.choices(all_samples, k=10)

    image_model_id = "Salesforce/blip-image-captioning-base"
    text = "a photography of"
    sample_processor = transformers.BlipProcessor.from_pretrained
    sample_model = transformers.BlipForConditionalGeneration.from_pretrained

    descriptions = ImageDescription(samples, image_model_id).generate_sample_descriptions(text, sample_processor, sample_model)
    print(descriptions)

    # dataset overall description
    dataset_input_text = generate_input_text(descriptions)
    print(dataset_input_text)

    dataset_tokenizer = transformers.T5Tokenizer.from_pretrained("google/flan-t5-base")
    dataset_model = transformers.T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    input_ids = dataset_tokenizer(dataset_input_text, return_tensors="pt").input_ids

    output = dataset_model.generate(input_ids, max_length=40).squeeze()
    dataset_description = dataset_tokenizer.decode(output)
    print(dataset_description)
