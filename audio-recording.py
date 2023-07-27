import sounddevice as sd
from scipy.io.wavfile import write
from transformers import Wav2Vec2Processor

processor_path = "./runs/wav2vec2-full"
processor = Wav2Vec2Processor.from_pretrained(processor_path)


# verify if sample rate of model corresponds to wav file sample rate
target_sample_rate = processor.feature_extractor.sampling_rate

sampleRate = 44100
seconds = 5

myRecording = sd.rec(int(seconds*sampleRate), samplerate=sampleRate, channels=1)
sd.wait()
write('output.wav', sampleRate, myRecording)