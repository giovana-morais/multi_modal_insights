import argparse
import config
import glob
import random

if __name__ == "__main__":

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
