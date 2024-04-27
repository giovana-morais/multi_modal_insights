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

    # FIXME: right now subfolders are being counted as files. this is not what i want.
    for current_dir, subdirs, files in os.walk(data_home):
        folder_structure[os.path.basename(current_dir)] = {}

        for f in files:
            ext = os.path.splitext(f)[-1]
            file_type_count.setdefault(ext, 0)
            file_type_count[ext] += 1

        if len(subdirs) != 0:
            folder_structure[os.path.basename(current_dir)]["folders"] = len(files)

        folder_structure[os.path.basename(current_dir)]["files"] = len(files)

    return folder_structure, file_type_count

def readme(dataset):
    test = glob.glob(os.path.join(dataset, "*README*"))
    test.append(glob.glob(os.path.join(dataset, "*readme*")))
    return test

if __name__ == "__main__":
    # dataset_path = sys.argv[1]
    dataset_path = "/home/gigibs/Documents/datasets/candombe"
    dataset = pathlib.Path(dataset_path)
    folder_structure, file_count = statistics(dataset)
    print("folder structure", folder_structure)
