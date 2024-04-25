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

def statistics(data_home):
    """
    """

    file_types = set()
    total_files = 0
    folder_structure = {}
    file_type_count = {}

    for current_dir, subdirs, files in os.walk(data_home):
        # folder_structure[current_dir] = {}
        print(f"{current_dir} \t {len(files)}")

        for f in files:
            ext = os.path.splitext(f)[-1]
            file_type_count.setdefault(ext, 0)
            file_type_count[ext] += 1

        folder_structure[current_dir] = len(files)


    print(file_type_count)
    return folder_structure, file_type_count

def splits():
    """
    Returns splits available or None.
    """

    return

def readme(dataset):
    test = glob.glob(os.path.join(dataset, "*README.md"))
    return test

if __name__ == "__main__":
    # dataset_path = sys.argv[1]
    dataset_path = "/home/gigibs/Documents/datasets/gtzan_genre"
    dataset = pathlib.Path(dataset_path)
    # readme(dataset)
    a = statistics(dataset)
    print("dataset type", a)
    b = readme(dataset)
    print("readme", b)

    # print(a)
