import transformers

# modality-specific models
# image
IMAGE_MODEL_ID = "Salesforce/blip-image-captioning-base"
IMAGE_PROCESSOR = transformers.BlipProcessor.from_pretrained
IMAGE_MODEL = transformers.BlipForConditionalGeneration.from_pretrained
IMAGE_SAMPLE_DESCRIPTION_TEXT = "a photography of"

# dataset description
DATASET_MODEL_ID = "google/flan-t5-base"
DATASET_TOKENIZER = transformers.T5Tokenizer.from_pretrained
DATASET_MODEL = transformers.T5ForConditionalGeneration.from_pretrained

# audio
AUDIO_PROMPT = "Is the audio sample music, environmental sound or speech? Can you
describe it?"
