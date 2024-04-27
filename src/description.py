import argparse
import config
import glob
import random

from audio_description import AudioDescription
from image_description import ImageDescription

def img_example():
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
    return descriptions


def music_example():
    # dataset sample descriptions
    all_samples = glob.glob("/home/gigibs/Documents/datasets/candombe/candombe_audio/*.flac")
    samples = random.choices(all_samples, k=10)

    descriptions = AudioDescription(
            samples=samples
    ).generate_sample_descriptions(config.IMAGE_SAMPLE_DESCRIPTION_TEXT)
    return descriptions

def envsound_example():
    return

if __name__ == "__main__":

    descriptions = music_example().generate_sample_descriptions(config.AUDIO_PROMPT)
    # combining sample descriptions to generate dataset description
    dds = DatasetDescription(
            descriptions=descriptions,
            tokenizer=config.DATASET_TOKENIZER,
            model=config.DATASET_MODEL,
            model_id=config.DATASET_MODEL_ID,
            verbose=True)

    dds.generate_description()
