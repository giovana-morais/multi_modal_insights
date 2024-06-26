import base64
import os
import glob
import zipfile

import datamapplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from bertopic import BERTopic
from bertopic.backend import MultiModalBackend
from bertopic.cluster import BaseCluster
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech, VisualRepresentation
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from huggingface_hub import login
from io import BytesIO
from torch import bfloat16
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoTokenizer, pipeline, AutoModelForCausalLM
from umap import UMAP

from key import hf_key

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


def set_topic_explainer_pipe():
    model_id = "llava-hf/llava-1.5-7b-hf"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_kwargs = {
            "quantization_config": quantization_config,
    }
    pipe = pipeline("image-to-text", model=model_id,  model_kwargs=model_kwargs)
    return pipe


def set_image_sumarizer_pipe():
    login(hf_key)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit quantization
        bnb_4bit_quant_type='nf4',  # Normalized float 4
        bnb_4bit_use_double_quant=True,  # Second quantization after the first
        bnb_4bit_compute_dtype=bfloat16  # Computation type
    )
    model_id = 'meta-llama/Llama-2-7b-chat-hf'

    # Llama 2 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Llama 2 Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
    )
    model.eval()

    # Our text generator
    generator = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1
    )
    return generator, tokenizer


def load_np_from_file(path):
     with open(path, 'rb') as f:
         return np.load(f)


def generate_topic_text(name,count):
    return f'group name: {name} | count: {count} images \n'


class DummyDimensionalityReductor:
  """
  Class that simulates a dimensionality reduction process for use pre-computed reduced embeddings on BerTopic
  """
  def __init__(self, reduced_embeddings):
    self.reduced_embeddings = reduced_embeddings


  def fit(self, X):
    return self


  def transform(self, X):
    return self.reduced_embeddings


class ImageTopicExtractor:
    """ Class that uses BerTopic to find Image clusters and describe them """

    def __init__(self,
                 img_list:list[str],
                 embedding_model:str = 'clip-ViT-B-32',
                 image_to_text_model="nlpconnect/vit-gpt2-image-captioning",
                 min_topic_size:float  = None,
                 dimensionality_reductor = None,
                 cluster_method = None,
                 clusters = None,
                 embeddings = None,
                 reduced_embeddings = None,
                 reduced_embeddings_2d = None,

                ):
        """
        arguments
        ---
            img_list         : list[str]
                list with path to imgs
            embedding_model       : name of embedding transformers model
            image_to_text_model: name of image_to_text transformers model
            min_topic_size    : min size of clusters
            dimensionality_reductor : Dimensionality Reduction technique. ignored if reduced_embeddings is not None
            cluster_method: Cluster technique. ignored if clusters is not None
            clusters: list of index identifying the cluster of each image. Can be a path to npy file or a list
            embeddings: precomputed embeddings.Can be a path to npy file or a list
            reduced_embeddings = precomputed reduced embeddings.Can be a path to npy file or a list
            reduced_embeddings_2d = precomputed reduced embeddings 2d used in visualizations.Can be a path to npy file or a list

        """
        self.img_list = img_list
        self.embedding_model = MultiModalBackend(embedding_model, batch_size=32)
        self.min_topic_size = max(1,int(0.1*len(img_list))) if min_topic_size is None else min_topic_size
        self.image_to_text_model = image_to_text_model
        self.hdbscan_model = cluster_method
        self.embeddings = load_np_from_file(embeddings) if type(embeddings) == str else embeddings
        self.reduced_embeddings = load_np_from_file(reduced_embeddings) if type(reduced_embeddings) == str else reduced_embeddings
        self.reduced_embeddings_2d = load_np_from_file(reduced_embeddings_2d) if type(reduced_embeddings_2d) == str else reduced_embeddings_2d
        self.clusters = load_np_from_file(clusters) if type(clusters) == str else clusters
        self.dimensionality_reductor = dimensionality_reductor if reduced_embeddings is None else DummyDimensionalityReductor(self.reduced_embeddings)
        self.hdbscan_model = cluster_method if clusters is None else BaseCluster()


    def fit(self, topic_explainer_pipe = False):
        """
        arguments
        ----
        topic_explainer_pipe: Hugging face pipeline used to create a readable description of each cluster. If false, No custom name will be added.
        """
        representation_model = {
            "Visual_Aspect": VisualRepresentation(image_to_text_model=self.image_to_text_model)
        }

        topic_model = BERTopic(
            # Pipeline models
            embedding_model=self.embedding_model,
            umap_model=self.dimensionality_reductor,
            hdbscan_model=self.hdbscan_model,
            representation_model=representation_model,
            min_topic_size = self.min_topic_size,

            verbose=True
        )
        self.topic_model = topic_model.fit(
                documents=None,
                images=self.img_list,
                embeddings=self.embeddings,
                y=self.clusters
        )

        if topic_explainer_pipe:
            self.update_labels_LLM(topic_explainer_pipe)
        return


    def describe_topic(self, topic, df, max_new_tokens, pipe, custom_prompt=False):
        """
        Internal function that uses a pipeline to create a readable description of each cluster
        """
        key_words = df[df.Topic == topic].Representation.to_list()[0]
        image = df[df.Topic == topic].Visual_Aspect.iloc[0]
        str_key_words = " "

        for word in key_words:
            str_key_words += f' {word}'

        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = f"USER: I have a group of images that contains the following sample images. \n <image>\n  The topic is described by the following keywords:{str_key_words} \n Based on the above    information,  give me a short label of the topic using no more than 10 words. Your description have to be true for all images at the same time.  \nASSISTANT:"
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
        response = outputs[0]["generated_text"]
        response = response.split("ASSISTANT:")[1]
        return response


    def update_labels_LLM(self, topic_explainer_pipe, prompt=False):
        """
        Internal function that updates the name of each cluster using describe topic functiom=n
        """
        df = self.get_topic_info()
        max_new_tokens = 200
        result = df.Topic.apply(lambda topic: self.describe_topic(topic,df,max_new_tokens,topic_explainer_pipe)).to_list()
        self.topic_model.set_topic_labels(result)
        return


    def get_topic_info(self):
        return self.topic_model.get_topic_info().copy()

    def display_topic_info_images(self, custom_labels=True):
        if custom_labels:
            df = self.topic_model.get_topic_info().drop(["Representative_Docs", "Name", "Representation"],axis=1)
        else:
            df = self.topic_model.get_topic_info().drop(["Representative_Docs", "Name"],axis=1)
        return df.to_html(formatters={'Visual_Aspect': image_formatter}, escape=False)

    def visualize_2d_clusters(self, hide_document_hover=True, custom_labels=True, hide_annotations=False):
        if self.reduced_embeddings_2d is None:
            embeddings = self.embedding_model.embed_images(images = self.img_list, verbose = True)
            vis = self.topic_model.visualize_documents(
                    self.img_list,
                    embeddings=embeddings,
                    hide_document_hover=hide_document_hover,
                    custom_labels=custom_labels,
                    hide_annotations=hide_annotations
            )
        else:
            vis = self.topic_model.visualize_documents(
                    self.img_list,
                    reduced_embeddings=self.reduced_embeddings_2d,
                    hide_document_hover=hide_document_hover,
                    custom_labels=custom_labels,
                    hide_annotations=hide_annotations
            )
        return vis

    def get_document_info(self):
        return self.topic_model.get_document_info(self.img_list)

    def get_final_description(self,dataset_summary_pipe, tokenizer, custom_labels = True):
        """
        Function that gives the final image_dataset description

        arguments
        ----
        dataset_summary_pipe = HF pipeline used to summarize the description
        tokenizer = tokenizer used on the HF pipeline
        custom_labels = True or False. If True, we use CustomName column as description of each cluster, otherwise we use name Column
        """

        df = self.get_topic_info()
        df = df[df.Topic>=0]

        if custom_labels:
            topic_texts = ''.join(df.apply(lambda x: generate_topic_text(x.CustomName, x.Count),axis=1).to_list())
        else:
            topic_texts = ''.join(df.apply(lambda x: generate_topic_text(x.Name, x.Count),axis=1).to_list())
        system_prompt = """
            <s>[INST] <<SYS>>
            You are a helpful, respectful and honest assistant for dataset description.
            <</SYS>>
            """

        example_prompt = """
        I have a dataset that contains the following group of images:
        group name:  Cars | count: 200 images
        group name:  Children. | count: 400 images
        group name:  People runing. | count: 300 images

        Based on the information above, please create a description of this group of images. This description should be in plain text, avoiding lists and using a common vocabulary.

        [/INST] This dataset comprises a diverse collection of images featuring various subjects. Within the dataset, there are 200 images of cars, capturing a range of makes, models, and settings. Additionally, there are 400 images focused on children, potentially depicting them engaged in various activities or scenarios. Furthermore, the dataset includes 300 images showcasing people running, suggesting a theme of physical activity or sports. Together, these images offer a multifaceted view of different subjects and activities, providing ample material for analysis and exploration.
        """

        main_prompt = """
        [INST]
        I have a dataset that contains the following group of images:
        [TOPICS]

        Based on the information about the topic above, please create a short description of this group of images.This description should be in plain text, and using a common vocabulary.
        [/INST]
        """

        prompt = system_prompt + example_prompt + main_prompt.replace("[TOPICS]",topic_texts)
        if(len(tokenizer.encode(prompt))> 4000):
            print('too much clusters, considering just top 10')
            df = df[df['Topic']<10]
            if(custom_labels):
                topic_texts = ''.join(df.apply(lambda x: generate_topic_text(x.CustomName, x.Count),axis=1).to_list())
            else:
                topic_texts = ''.join(df.apply(lambda x: generate_topic_text(x.Name, x.Count),axis=1).to_list())
            prompt = system_prompt + example_prompt + main_prompt.replace("[TOPICS]",topic_texts)
        res = dataset_summary_pipe(prompt)
        res = res[0]["generated_text"]
        res = res.split('[/INST]')[2]

        return res

    def datamap_plot(self,
                    title = "Image Clusters",
                    subtitle = None,
                    show = True,
                    save_path = None,
                    custom_labels = True):

        if custom_labels:
            labels = self.topic_model.get_document_info(self.img_list).CustomName.to_list()
        else:
            labels = self.topic_model.get_document_info(self.img_list).Name.to_list()

        if self.reduced_embeddings_2d is None:
            embeddings = self.embedding_model.embed_images( images = self.img_list, verbose = True)
            reduced_embeddings_2d = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

        else:
            reduced_embeddings_2d = self.reduced_embeddings_2d

        # Run the visualization
        datamapplot.create_plot(
            reduced_embeddings_2d,
            labels,

            use_medoids=True,

            figsize=(12, 12),

            dpi=100,

            title=title,
            # Universally set a font family for the plot.
            fontfamily="Roboto",

            # Takes a dictionary of keyword arguments that is passed through to
            # matplotlib’s 'title' 'fontdict' arguments.
            title_keywords={
                "fontsize":36,
                "fontfamily":"Roboto Black"
            },
            # Takes a dictionary of keyword arguments that is passed through to


            # By default DataMapPlot tries to automatically choose a size for the text that will allow
            # all the labels to be laid out well with no overlapping text. The layout algorithm will try
            # to accommodate the size of the text you specify here.
            label_font_size=8,
            label_wrap_width=16,
            label_linespacing=1.25,
            # Default is 1.5. Generally, the values of 1.0 and 2.0 are the extremes.
            # With 1.0 you will have more labels at the top and bottom.
            # With 2.0 you will have more labels on the left and right.
            label_direction_bias=1.3,
            # Controls how large the margin is around the exact bounding box of a label, which is the
            # bounding box used by the algorithm for collision/overlap detection.
            # The default is 1.0, which means the margin is the same size as the label itself.
            # Generally, the fewer labels you have the larger you can make the margin.
            label_margin_factor=2.0,
            # Labels are placed in rings around the core data map. This controls the starting radius for
            # the first ring. Note: you need to provide a radius in data coordinates from the center of the
            # data map.
            # The defaul is selected from the data itself, based on the distance from the center of the
            # most outlying points. Experiment and let the DataMapPlot algoritm try to clean it up.
            label_base_radius=15.0,

            # By default anything over 100,000 points uses datashader to create the scatterplot, while
            # plots with fewer points use matplotlib’s scatterplot.
            # If DataMapPlot is using datashader then the point-size should be an integer,
            # say 0, 1, 2, and possibly 3 at most. If however you are matplotlib scatterplot mode then you
            # have a lot more flexibility in the point-size you can use - and in general larger values will
            # be required. Experiment and see what works best.
            point_size=4,

            # Market type. There is only support if you are in matplotlib's scatterplot mode.
            # https://matplotlib.org/stable/api/markers_api.html
            marker_type="o",

            arrowprops={
                "arrowstyle":"wedge,tail_width=0.5",
                "connectionstyle":"arc3,rad=0.05",
                "linewidth":0,
                "fc":"#33333377"
            },

            add_glow=True,
            # Takes a dictionary of keywords that are passed to the 'add_glow_to_scatterplot' function.
            glow_keywords={
                "kernel_bandwidth": 0.75,  # controls how wide the glow spreads.
                "kernel": "cosine",        # controls the kernel type. Default is "gaussian". See https://scikit-learn.org/stable/modules/density.html#kernel-density.
                "n_levels": 32,            # controls how many "levels" there are in the contour plot.
                "max_alpha": 0.9,          # controls the translucency of the glow.
            },

            darkmode=False,
        )
        if(show):
            plt.tight_layout()
        if(save_path):
            # Save the plot
            plt.savefig(path)
        return
