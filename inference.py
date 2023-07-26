from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch, torchaudio, os

# Change model_path accordingly when using either full or quantized model
model_path = "./runs/wav2vec2-full"

model = AutoModelForCTC.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(model_path)


# verify if sample rate of model corresponds to wav file sample rate
target_sample_rate = processor.feature_extractor.sampling_rate
waveform, file_sample_rate = torchaudio.load("./audio/input.wav", normalize=True) 

if file_sample_rate!= target_sample_rate:
    resampler= torchaudio.transforms.Resample(file_sample_rate, target_sample_rate)
    waveform = resampler(waveform)


#performs inference on batch and returns results
def map_to_result(batch_tensor, model):
    
    # torch.no_grad() ensures no gradients are changed during inferencing
    with torch.no_grad():
        logits = model(batch_tensor.unsqueeze(0)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    predicted_text = processor.batch_decode(pred_ids)[0]

    return predicted_text


# Preprocess input data
input = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt", padding=True)
input_tensor = input["input_values"].squeeze()


predicted_text = map_to_result(input_tensor, model)
print(predicted_text)
