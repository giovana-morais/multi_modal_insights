# multi_modal_insights

## how it works
We detect the modality of the data by analyzing file types

![big_data_workflow](https://github.com/giovana-morais/multi_modal_insights/assets/12520431/a4cc1d34-ad45-4976-a7d4-6b3bebb5c213)

## performance

| Dataset             | Type    | Size                                   | Time              |
|---------------------|---------|----------------------------------------|-------------------|
| 20news              | text    | 18000 documents                        | 79 s              |
| Flicker8k           | image   | 8091 samples                           | 732,5 s (~12 min) |
| Candombe            | audio   | 35 songs                               | 441 s (~7 min)    |
| Coco 2014           | image   | 20000 images sampled from 82783 images | 1387.5s (~23 min) |
| Smoking Health Data | tabular | 3900 rows                              | 14s               |

## how to use

Clone the repo, `cd` into it and install all the dependencies
`pip install -r requirements.txt`

you have different modalities examples in the `example_description.py` script.
choose your favorite modality and see the project in action with a sample
dataset.

## folder structure

* `src`: source code
* `notebooks`: examples of single-modality descriptions
* `slurm_jobs`: SLURM jobs to run the code on [HPC](https://sites.google.com/nyu.edu/nyu-hpc/home?authuser=0)

## team behind it

We started this project as a final project for the Big Data course taught by
Prof. Juliana Freire at NYU Tandon.

[Felipe Oliveira](https://github.com)
[Giovana Morais](https://github.com/giovana-morais)
[Roman Vakhrushev](https://github.com/golcz)

Feedbacks are appreciated!
