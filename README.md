# Robot Automatic Speech Recognition using Wav2vec2

## What is this?

This Automatic Speech Recognition (ASR) model uses pretrained Wav2Vec2 - a Deep Learning Audio model introduced by Facebook AI Research, which is further finetuned on Timit - an acoustic speech dataset, in order to increase the ASR's effectiveness in speech recognition 

## What is it used for?

The Robot-ASR model is intended to be deployed on Trinity Robotics' Storm robot in order to give operation commands - like "go left" for example, using audio input. However, this model is not limited to operations on Storm and can be deployed on any device that supports the necessary requirements. The model can also be quantized to reduce the file size and better suit operations on edge or mobile devices with limited storage space.

## Performance

When evaluating the performance of this ASR model, the Robot ASR base and quantized variations were tested against the non-finetuned Wav2Vec2 base model. All three models were evaluated using a Timit test set of 1680 audio samples with Word Error Rate (WER) being used evaluate each model's accuracy. 

Models | WER | Avg Memory Usage, Gb | Inference Time, s | 
--- | --- | --- | --- | --- |
Robot-ASR | 0.277 | 5.38 | 48 | 
--- | --- | --- | --- | --- |
Wav2Vec2-base| 0.360 | 6.68 | 35 | 

## Resources and References
https://www.trinityrobotics.ca

https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb#scrollTo=Adm0LngNNxq7

https://arxiv.org/pdf/2006.11477.pdf
