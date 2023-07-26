from transformers import AutoModelForCTC, Wav2Vec2Processor
import os

# Change model_path accordingly when using either full or quantized model
model_path = "./runs/wav2vec2-full"

# first time users should download the model instead
model_download = "patrickvonplaten/wav2vec2-base-timit-demo-google-colab"

model = AutoModelForCTC.from_pretrained(model_download)
processor = Wav2Vec2Processor.from_pretrained(model_download)


if os.path.exists(model_path) == False:
    os.mkdir(model_path)
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)