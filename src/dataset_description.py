from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer, OnlineCountVectorizer
from hdbscan import HDBSCAN
from huggingface_hub import login
from transformers import BitsAndBytesConfig, AutoTokenizer, pipeline, AutoModelForCausalLM
from torch import bfloat16
from umap import UMAP

from .key import hf_key

__MODEL_ID__ = "meta-llama/Llama-2-7b-chat-hf"
__TOKENIZER__ = AutoTokenizer.from_pretrained
__MODEL__ = AutoModelForCausalLM.from_pretrained

class DatasetDescription:
    def __init__(self, **kwargs):
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
        self.hdbscan_model = HDBSCAN(
                min_cluster_size=15,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
        )
        self.vectorizer_model = OnlineCountVectorizer()
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    def dataset_description(self, prompt):
        """
        given a prompt filled with topic texts, creates a summary description
        """
        print("[INFO] creating dataset description")
        login(hf_key)
        tokenizer = __TOKENIZER__(__MODEL_ID__)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type='nf4',  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=bfloat16  # Computation type
        )

        model = __MODEL__(
            __MODEL_ID__,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map='auto',
        )
        model.eval()

        generator = pipeline(
                model=model,
                tokenizer=tokenizer,
                task='text-generation',
                temperature=0.1,
                max_new_tokens=500,
                repetition_penalty=1.1
        )
        output = generator(prompt)
        output = output[0]["generated_text"]
        output = output.split('[/INST]')[-1]

        return output

    def topic_model(self, embedding_model, representation_model=None):
        """
        given an embedding model and an optional represetation model, returns
        the BERTopic configurations
        """
        model = BERTopic(
            embedding_model=embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=representation_model
        )

        return model

    def create_prompt(topic_texts):
        raise NotImplementedError

    def generate_topic_text(name, count):
        raise NotImplementedError
