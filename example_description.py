"""
Usage
    python example_description.py
"""
import argparse
import glob
import os
import random

from src.text_description import TextDescription
from src.image_description import ImageDescription
from src.audio_description import AudioDescription
from src.tabular_description import TabularDescription

def text_example():
    from sklearn.datasets import fetch_20newsgroups

    data_home = "sample_data/text/"
    data = fetch_20newsgroups(data_home=data_home, subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    dataset = TextDescription(data=data)
    return dataset.dataset_description

def image_example():
    from sentence_transformers import util
    img_folder = 'sample_data/image/photos/'

    # download data if folder does not exist or if it is empty
    if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
        print("[INFO] Downloading Flicker8k data")
        os.makedirs(img_folder, exist_ok=True)

        if not os.path.exists('Flickr8k_Dataset.zip'):   #Download dataset if does not exist
            util.http_get('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip', 'Flickr8k_Dataset.zip')

            for folder, file in [(img_folder, 'Flickr8k_Dataset.zip')]:
                with zipfile.ZipFile(file, 'r') as zf:
                    for member in tqdm(zf.infolist(), desc='Extracting'):
                        zf.extract(member, folder)

    data = list(glob.glob('sample_data/image/photos/Flicker8k_Dataset/*.jpg'))
    dataset = ImageDescription(data=data)

    return dataset.dataset_description

def hpc_image_example():
    data = list(glob.glob("/coco/train2014/*.jpg"))
    dataset = ImageDescription(data=data)

    return dataset.dataset_description

def audio_example():
    # audio_folder = "/media/gigibs/DD02EEEC68459F17/datasets/candombe/candombe_audio/*.flac"
    audio_folder = "/scratch/work/marl/datasets/mir_datasets/candombe/audio/*.wav"

    data = list(glob.glob(audio_folder))
    print(len(data))
    dataset = AudioDescription(data=data)

    return dataset.dataset_description

def tabular_example():
    data = "sample_data/tabular/smoking_health_data_final.csv"
    dataset = TabularDescription(data_home=data)

    return dataset.dataset_description

if __name__ == "__main__":

    import time
    # print("Generating image description")
    # start = time.time()
    # print(hpc_image_example())
    # print(f"Duration: {time.time() - start}")

    print("Generating text description")
    start = time.time()
    print(text_example())
    print(f"Duration: {time.time() - start}")

    print("Generating image description")
    start = time.time()
    print(image_example())
    print(f"Duration: {time.time() - start}")

    print("Generating audio description")
    start = time.time()
    print(audio_example())
    print(f"Duration: {time.time() - start}")

    print("Generating tabular description")
    start = time.time()
    print(tabular_example())
    print(f"Duration: {time.time() - start}")
