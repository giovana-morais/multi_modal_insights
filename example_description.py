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
# from src.tabular_description import TabularDescription

def text_example():
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    dataset = TextDescription(data=data)
    return dataset.dataset_description

def image_example():
    img_folder = 'sample_data/image/photos/'

    # download flickrdata if folder does not exist or if it is empty
    if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
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

def audio_example():
    audio_folder = "/media/gigibs/DD02EEEC68459F17/datasets/candombe/candombe_audio/*.flac"

    data = list(glob.glob(audio_folder))
    print(len(data))
    dataset = AudioDescription(data=data)

    return dataset.dataset_description

if __name__ == "__main__":

    # print(text_example())
    # print(image_example())

    print(audio_example())
