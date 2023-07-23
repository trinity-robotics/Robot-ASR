from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch, torchaudio

# 1. Instantiate the processor and model classes.
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 2. Load the model weights from the .bin file.
model_path = "./runs/jul21/pytorch_model.bin"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# 3. Now the 'model' object contains the loaded weights and is ready for inference or fine-tuning.

waveform, file_sample_rate = torchaudio.load("./audio/output.wav")
waveform = waveform.squeeze(0)

#print(file_sample_rate)

target_sample_rate = processor.feature_extractor.sampling_rate

#print(target_sample_rate)

if file_sample_rate!= target_sample_rate:
    resampler= torchaudio.transforms.Resample(file_sample_rate, target_sample_rate)
    waveform = resampler(waveform)
#Resample wavfile according to model's sample rate

newInput = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt", padding=True)


# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#print("Shape here", newInput.input_values.to(device))
# Pass the preprocessed data through the model to obtain predictions
with torch.no_grad():
    logits = model(newInput.input_values.to(device)).logits

# Post-process the model's output to obtain the predicted text
pred_ids = torch.argmax(logits, dim=-1)
predicted_text = processor.batch_decode(pred_ids)[0]

print("Predicted Text:", predicted_text)
