import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load the pre-trained model and tokenizer
model_name = 'facebook/wav2vec2-base-960h'
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

# Example audio file path
audio_file = 'audio/chatbot_output_waveRNN.wav'

# Load and preprocess the audio file
input_audio, sample_rate = sf.read(audio_file)
input_values = tokenizer(input_audio, return_tensors='pt').input_values

# Perform inference
with torch.no_grad():
    logits = model(input_values).logits

# Decode the predicted transcription
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.decode(predicted_ids[0])

# Print the predicted transcription
print("Predicted Transcription:", transcription)
