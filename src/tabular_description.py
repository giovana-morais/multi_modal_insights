import random

import datamart_profiler
import pandas

from .dataset_description import DatasetDescription

__NUM_SAMPLES__ = 10

class TabularDescription(DatasetDescription):
    def __init__(self, data_home=None):
        super(TabularDescription, self).__init__()
        if data_home is None:
            raise ValueError("No data path was provided")

        self.data = data_home
        self.profile = datamart_profiler.process_dataset(self.data)
        self.sample_data = self.get_sample()
        self.prompt = self.create_prompt()
        self.dataset_description = super().dataset_description(self.prompt)

        return

    def get_sample(self):
        df = pandas.read_csv(self.data)
        sample = df.sample(__NUM_SAMPLES__)
        return sample


    def create_prompt(self):
        system_prompt = """
            <S>[INST] <<SYS>>
            you are a helpful, respectful and honest assistant for dataset description.
            <</SYS>>
            """

        main_prompt = """
        [INST]
        You are given the following file JSON file, which represents a profile of a tabular dataset:
        [PROFILE]

        The rows look like this:

        [SAMPLE_ROWS]

        Write a description for the whole dataset, do not mention what was provided to you for this task. This description should be in plain text, using a common vocabulary.
        [/INST]
        """

        main_prompt = main_prompt.replace("[PROFILE]", str(self.profile))
        main_prompt = main_prompt.replace("[SAMPLE_ROWS]", self.sample_data.to_string(index=False))
        prompt = system_prompt + main_prompt

        return prompt
