"""
Usage
    python example_description.py --data_home "../datasets/candombe/candombe_audio/*.flac"
"""
import argparse
import glob
import os
import random

import src.config as config
from src.audio_description import AudioDescription
from src.image_description import ImageDescription


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_home", type=str, help="path to your dataset",
            required=True)
    # parser.add_argument("--modality", type="str", choices=["audio", "img",
    #     "text", "tabular"], help="data modality", required=False)

    args = parser.parse_args()
    # data_home = "/home/gigibs/Documents/datasets/candombe/candombe_audio"

    all_samples = glob.glob(os.path.join(args.data_home, "*.flac"))
    samples = random.choices(all_samples, k=10)
    print(samples)

    audio_dataset_description = AudioDescription(
            data_home=args.data_home,
            samples=samples
    ).dataset_description
    print(audio_dataset_description)
