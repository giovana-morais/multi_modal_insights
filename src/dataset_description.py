import transformers

__DATASET_MODEL_ID__ = "google/flan-t5-base"
__DATASET_TOKENIZER__ = transformers.T5Tokenizer.from_pretrained
__DATASET_MODEL__ = transformers.T5ForConditionalGeneration.from_pretrained

class DatasetDescription:
    def __init__(self):
        self.dataset_tokenizer = __DATASET_TOKENIZER__(__DATASET_MODEL_ID__)
        self.dataset_model = __DATASET_MODEL__(__DATASET_MODEL_ID__)

    def generate_input_text(self, descriptions, verbose=False):
        dataset_input_text = f"The following items are descriptions of {len(self.descriptions)} items of the same dataset. Create a unique description of the whole dataset for me that summarizes the common characteristics."

        for idx, desc in enumerate(self.descriptions):
            d = f"\n{idx+1})  {desc}"
            dataset_input_text += d

        dataset_input_text += "\n"

        if verbose:
            print(dataset_input_text)
        return dataset_input_text

    def dataset_description(self, descriptions):
        dataset_input_text = self.generate_input_text(descriptions)

        input_ids = self.dataset_tokenizer(dataset_input_text, return_tensors="pt").input_ids

        output = self.dataset_model.generate(input_ids, max_length=200).squeeze()
        description = self.dataset_tokenizer.decode(output)

        return description

    # def dataset_metadata(self):
    #     """
    #     Based on data_home, get

    #     """
    #     raise NotImplementedError
