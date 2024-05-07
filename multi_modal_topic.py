from bertopic import BERTopic
from bertopic.representation import LlamaCPP
from bertopic.vectorizers import ClassTfidfTransformer, OnlineCountVectorizer
from hdbscan import HDBSCAN
from llama_cpp import Llama
from umap import UMAP


def get_sample_dataset(modality):
    if modality == "text":
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    elif modality == "image":
        raise NotImplementedError
    elif modality == "audio":
        raise NotImplementedError
    elif modality == "tabular":
        raise NotImplementedError

    return data


def create_prompt(topic_texts):
    system_prompt = """
        <S>[INST] <<SYS>>
        you are a helpful, respectful and honest assistant for dataset description.
        <</SYS>>
        """

    example_prompt = """
    Q: I have a dataset that contains the following group of images:
    group name:  Cars | count: 200 docs
    group name:  Children. | count: 400 docs
    group name:  People runing. | count: 300 docs

    Based on the information above, please create a description of this group of images. This description should be in plain text, avoiding lists and using a common vocabulary.

    A: This dataset comprises a diverse collection of images featuring various subjects. Within the dataset, there are 200 images of cars, capturing a range of makes, models, and settings. Additionally, there are 400 images focused on children, potentially depicting them engaged in various activities or scenarios. Furthermore, the dataset includes 300 images showcasing people running, suggesting a theme of physical activity or sports. Together, these images offer a multifaceted view of different subjects and activities, providing ample material for analysis and exploration.
    """

    main_prompt = """
    Q:
    I have a dataset that contains the following group of images:
    [TOPICS]

    Based on the information about the topic above, please create a short description of this group of images.This description should be in plain text, and using a common vocabulary.
    A:
    """

    main_prompt = main_prompt.replace("[TOPICS]", topic_texts)

    prompt = system_prompt + example_prompt + main_prompt
    return prompt


def generate_topic_text(name, count, modality_ext):
    return f'group name: {name} | count: {count} {modality_ext}\n'


if __name__ == "__main__":
    modality_config = {
        "text": {
            "embedding_model": "all-MiniLM-L6-v2",
            "modality_ext": "docs",
            # "representation_model": LlamaCPP("models/llama-2-7b.Q5_K_S.gguf")
        },
        "image": {
            "embedding_model": "clip-ViT-B-32",
            "representation_model": "nlpconnect/vit-gpt2-image-captioning",
            "modality_ext": "images",
        },
        "audio": {
            "embedding_model" : "",
            "modality_ext": "files"
        },
        "tabular": {
            "embedding_model" : None,
        }
    }

    # assume our datatype is text
    modality = "text"
    data = get_sample_dataset(modality)

    configs = modality_config[modality]

    # 1. embeddings
    embedding_model = configs["embedding_model"]

    # 2. dimensionality reduction
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
    metric="cosine")

    # 3. clustering
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean',
            cluster_selection_method='eom', prediction_data=True)

    # 4. vectorizers
    vectorizer_model = OnlineCountVectorizer()

    # 5. c-TF-IDF
    small_data = False
    if small_data:
        ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    else:
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        # representation_model=configs["representation_model"],
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model
    )

    topic_model.fit(data)
    topics = topic_model.get_topic_info()

    # 6. fine-tune topics (optional)
    # here is where we put our summarizer, i.e. dataset descriptor

    # -1 are outliers, so we remove them
    # and get the top 20 topics
    topics = topics[topics["Topic"] > -1].head(20)
    topic_text = ''.join(topics.apply(lambda x: generate_topic_text(x.Name,
        x.Count, configs["modality_ext"]), axis=1).to_list())

    prompt = create_prompt(topic_text)

    llm = Llama(
        model_path="models/llama-2-7b.Q5_K_S.gguf",
        n_ctx=8192,
        n_batch=512
    )

    output = llm(
        prompt,
        max_tokens=-1,
        echo=False,
        temperature=0.2,
        top_p=0.1
    )

    # output = output[0]["generated_text"]
    # print(output)
    # output = output.split('[/INST]')[2]
