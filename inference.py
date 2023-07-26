from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch, torchaudio

# Replace the model identifier with the desired model
model_path = "./runs/jul22"
model = AutoModelForCTC.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(model_path)


target_sample_rate = processor.feature_extractor.sampling_rate
waveform, file_sample_rate = torchaudio.load("./audio/input.wav", normalize=True) 


if file_sample_rate!= target_sample_rate:
    resampler= torchaudio.transforms.Resample(file_sample_rate, target_sample_rate)
    waveform = resampler(waveform)
#Resample wavfile according to model's sample rate


#performs inference on batch and returns results
def map_to_result(batch_tensor, model):
    
    # torch.no_grad() ensures no gradients are changed during inferencing
    with torch.no_grad():
        logits = model(batch_tensor.unsqueeze(0)).logits
        # Since we are using raw waveform as input, no need to process through the processor

    pred_ids = torch.argmax(logits, dim=-1)
    predicted_text = processor.batch_decode(pred_ids)[0]

    return predicted_text


#preprocess input data
input = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt", padding=True)
input_tensor = input["input_values"].squeeze()
print("input tensor: ", input_tensor)

predicted_text = map_to_result(input_tensor, model)

print(predicted_text)
