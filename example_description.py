"""
Usage
    python example_description.py --data_home "../datasets/candombe/candombe_audio/*.flac"
"""
import argparse
import glob
import os
import random

from src.text_description import TextDescription
from src.image_description import ImageDescription
from src.audio_description import AudioDescription
from src.tabular_description import TabularDescription

if __name__ == "__main__":
    # text example
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    text_dataset = TextDescription(data=data)
    print(text_dataset.dataset_description)

    # image example


    # all_samples = glob.glob(os.path.join(args.data_home, "*.flac"))
    # samples = random.choices(all_samples, k=10)
    # print(samples)

    # audio_dataset_description = AudioDescription(
    #         data_home=args.data_home,
    #         samples=samples
    # ).dataset_description
    # print(audio_dataset_description)
