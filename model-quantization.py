from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch, torchaudio, os

# Replace the model identifier with the desired model
model_path = "./runs/wav2vec2-trained"

model = AutoModelForCTC.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(model_path)


#   Changing file sampling rate

target_sample_rate = processor.feature_extractor.sampling_rate
# Load the .wav file using torchaudio or any other library
waveform, file_sample_rate = torchaudio.load("./audio/chatbot_output_waveRNN.wav", normalize=True)  # Replace with the path to your .wav file
# Make sure the sample rate of the loaded waveform matches the model's expected sample rate

#print(target_sample_rate)

if file_sample_rate!= target_sample_rate:
    resampler= torchaudio.transforms.Resample(file_sample_rate, target_sample_rate)
    waveform = resampler(waveform)
#Resample wavfile according to model's sample rate

#preprocess input data
input = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt", padding=True)
input_tensor = input["input_values"].squeeze()
print("input tensor: ", input_tensor)


torch.backends.quantized.engine = 'qnnpack'
#bug fix for torch quantize issue: https://github.com/pytorch/pytorch/issues/29327

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)


# Saving the now quantized model

save_directory = 'wav2vec2-quantized'
parent_directory = './runs'

quantized_model_path = os.path.join(parent_directory, save_directory)
if os.path.exists(quantized_model_path) == False:
    os.mkdir(quantized_model_path)

traced_model = torch.jit.trace(quantized_model, input_tensor.unsqueeze(0), strict=False)
torch.jit.save(traced_model, "./runs/wav2vec2-quantized/wav2vec2_trace.pt")

