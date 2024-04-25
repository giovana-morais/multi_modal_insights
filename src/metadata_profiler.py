"""
Generates metadata of a given dataset

---
Usage

metadata_profiler.py --path /MillionSongDataset

"""
import glob
import os
import pathlib
import sys

def num_files(dataset):
    """
    """

    folder_structure = {}
    for current_dir, subdirs, files in os.walk(dataset):
        # folder_structure[current_dir] = {}
        total_files = 0

        print(f"{current_dir} \t {len(files)}")

        # for dirname in subdirs:
        #     print("\t" + dirname)

        folder_structure[current_dir] = len(files)
    return folder_structure

def splits():
    """
    Returns splits available or None.
    """

    return

def data_type():
    """
    Returns types of files found in the dataset folder
    """
    return

def structure_output_text():
    return

def readme(dataset):
    test = glob.glob("*README.*")
    print(test)
    return test

if __name__ == "__main__":
    # dataset_path = sys.argv[1]
    dataset_path = "/home/gigibs/Documents/datasets/gtzan_genre"
    dataset = pathlib.Path(dataset_path)
    # readme(dataset)
    a = num_files(dataset)

    # print(a)
