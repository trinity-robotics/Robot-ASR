from transformers import AutoModelForCTC, Wav2Vec2Processor, AutoFeatureExtractor
import torch, torchaudio

# Replace the model identifier with the desired model
model_identifier = "patrickvonplaten/wav2vec2-base-timit-demo-google-colab"
model_path = "./runs/jul22"

# Replace the tokenizer identifier with the desired tokenizer
processor_identifier = "patrickvonplaten/wav2vec2-base-timit-demo-google-colab"
processor_path = "./runs/jul22"

# Manually download the model and tokenizer files
model = AutoModelForCTC.from_pretrained(model_identifier)
model.save_pretrained(model_path)

processor = Wav2Vec2Processor.from_pretrained(processor_identifier)
processor.save_pretrained(processor_path)

def map_to_result(batch):
    # torch.no_grad() ensures no gradients are changed during inferencing
    with torch.no_grad():
        input_values = batch["input_values"].squeeze()  # Remove the extra dimension
        # Since we are using raw waveform as input, no need to process through the processor
        logits = model(input_values.unsqueeze(0)).logits
        
        #by removing torch.tensor(), it ensured that the tensor's shape is 

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    #batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch


#   Changing file sampling rate

target_sample_rate = processor.feature_extractor.sampling_rate
# Load the .wav file using torchaudio or any other library
waveform, file_sample_rate = torchaudio.load("./audio/output.wav", normalize=True)  # Replace with the path to your .wav file
# Make sure the sample rate of the loaded waveform matches the model's expected sample rate

#print(target_sample_rate)

if file_sample_rate!= target_sample_rate:
    resampler= torchaudio.transforms.Resample(file_sample_rate, target_sample_rate)
    waveform = resampler(waveform)
#Resample wavfile according to model's sample rate

#preprocess input data
input = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt", padding=True)



# Run inference using the map_to_result function
result = map_to_result(input)

# The predicted text is stored in the "pred_str" key of the result dictionary
predicted_text = result["pred_str"]

print("Predicted Text:", predicted_text)
'''
# Now, you can use the model and tokenizer offline
model = AutoModelForCTC.from_pretrained(model_path)
tokenizer = AutoFeatureExtractor.from_pretrained(tokenizer_path)
'''