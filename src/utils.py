import os
import glob
import zipfile
import numpy as np
import pandas as pd
import base64
import datamapplot
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import HTML
from tqdm import tqdm
from sentence_transformers import util
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech, VisualRepresentation
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.cluster import BaseCluster
from bertopic.backend import MultiModalBackend

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

class Dummy_Dimensionality_Reductor:
  """ Class that simulates a dimensionality reduction process for use pre-computed reduced embeddings on BerTopic """
  def __init__(self, reduced_embeddings):
    self.reduced_embeddings = reduced_embeddings

  def fit(self, X):
    return self

  def transform(self, X):
    return self.reduced_embeddings

def load_np_from_file(path):
     with open(path, 'rb') as f:
         return np.load(f)
class Image_Topic_Extractor:
    """ Class  that uses BerTopic to find Image clusters and describe them """

    def __init__(self,
                 img_list:list[str],
                 embedding_model:str = 'clip-ViT-B-32',
                 image_to_text_model="nlpconnect/vit-gpt2-image-captioning",
                 min_topic_size:float  = None,
                 Dimensionality_reductor = None,
                 Cluster_method = None,
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
            Dimensionality_reductor : Dimensionality Reduction technique. ignored if reduced_embeddings is not None
            Cluster_method: Cluster technique. ignored if clusters is not None
            clusters: list of index identifying the cluster of each image. Can be a path to npy file or a list
            embeddings: precomputed embeddings.Can be a path to npy file or a list
            reduced_embeddings = precomputed reduced embeddings.Can be a path to npy file or a list
            reduced_embeddings_2d = precomputed reduced embeddings 2d used in visualizations.Can be a path to npy file or a list
            
        """
        self.img_list = img_list
        self.embedding_model = MultiModalBackend(embedding_model, batch_size=32)
        self.min_topic_size = max(1,int(0.1*len(img_list))) if min_topic_size is None else min_topic_size
        self.image_to_text_model = image_to_text_model
        self.hdbscan_model = Cluster_method
        self.embeddings = load_np_from_file(embeddings) if type(embeddings) == str else embeddings
        self.reduced_embeddings = load_np_from_file(reduced_embeddings) if type(reduced_embeddings) == str else reduced_embeddings
        self.reduced_embeddings_2d = load_np_from_file(reduced_embeddings_2d) if type(reduced_embeddings_2d) == str else reduced_embeddings_2d
        self.clusters = load_np_from_file(clusters) if type(clusters) == str else clusters
        self.Dimensionality_reductor = Dimensionality_reductor if reduced_embeddings is None else Dummy_Dimensionality_Reductor(self.reduced_embeddings)
        self.hdbscan_model = Cluster_method if clusters is None else BaseCluster()
        

    def fit(self):
        representation_model = {
                "Visual_Aspect": VisualRepresentation(image_to_text_model=self.image_to_text_model)
        }
        topic_model = BERTopic(

        # Pipeline models
        embedding_model=self.embedding_model,
        umap_model=self.Dimensionality_reductor,
        hdbscan_model=self.hdbscan_model,
        representation_model=representation_model,
        
        verbose=True
    )
        self.topic_model = topic_model.fit(documents = None, images = self.img_list, embeddings=self.embeddings, y=self.clusters)
        return
    def get_topic_info(self):
        return self.topic_model.get_topic_info()

    def display_topic_info_images(self):
        # Extract dataframe
        df = self.topic_model.get_topic_info().drop(["Representative_Docs", "Name"],axis=1)
        HTML(df.to_html(formatters={'Visual_Aspect': image_formatter}, escape=False))
        return HTML(df.to_html(formatters={'Visual_Aspect': image_formatter}, escape=False))

    def visualize_2d_clusters(self,hide_document_hover=True):
        return self.topic_model.visualize_documents(self.img_list,
                                                    reduced_embeddings=self.reduced_embeddings_2d,
                                                    hide_document_hover=hide_document_hover)
    
    def get_document_info(self): 
        return self.topic_model.get_document_info(self.img_list)
    def datamap_plot(self, 
                    title = "Image Clusters",
                    subtitle = None,
                    show = True,
                    save_path = None):
             
        # Run the visualization
        datamapplot.create_plot(
            self.reduced_embeddings_2d,
            self.topic_model.get_document_info(self.img_list).Name.to_list(),
        
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



        

        