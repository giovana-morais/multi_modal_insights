class DatasetDescription:
    def __init__(self, descriptions, tokenizer, model, model_id, verbose=False):
        self.descriptions = descriptions
        self.tokenizer = tokenizer(model_id)
        self.model = model(model_id)
        self.verbose = verbose
        return

    def generate_input_text(self, verbose=False):
        dataset_input_text = f"The following items are descriptions of {len(self.descriptions)} items of the same dataset. Create a unique description of the whole dataset for me that summarizes the common characteristics."

        for idx, desc in enumerate(self.descriptions):
            d = f"\n{idx+1})  {desc}"
            dataset_input_text += d

        dataset_input_text += "\n"

        if self.verbose:
            print(dataset_input_text)
        return dataset_input_text

    def generate_description(self):
        dataset_input_text = self.generate_input_text()

        input_ids = self.tokenizer(dataset_input_text, return_tensors="pt").input_ids

        output = self.model.generate(input_ids, max_length=200).squeeze()
        dataset_description = self.tokenizer.decode(output)

        if self.verbose:
            print(dataset_description)
        return dataset_description
