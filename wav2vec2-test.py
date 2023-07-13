from datasets import load_dataset, Audio

#need to pip install soundfile and librosa 

minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
minds = minds.train_test_split(test_size=0.2)
#print(minds)

minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
#print(minds["train"][0])

#from huggingface_hub import notebook_login
#notebook_login()
#for notebook use


# pre-processing, preparing data and model for training

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
#initializes tokenizer

minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
#Changing sampling rate to 16 000 for wav2vec2 
#print(minds["train"][0])

def uppercase(example):
    return {"transcription": example["transcription"].upper()}

minds = minds.map(uppercase)
#Coverting all text to lower case

def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch

encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

#data collators used to form batches using list of dataset elements 
@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")



# evaluate, preparing metrics for tracking training progression

import evaluate
wer = evaluate.load("wer")

import numpy as np

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# training 

from transformers import AutoModelForCTC, TrainingArguments, Trainer

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# Auto classes are functions that automatically retrieves desired model based on its name and path
# In this case, it is looking for a model that is alse compatible with CTC loss

'''

#warmup steps used 

training_args = TrainingArguments(
    output_dir="my_awesome_asr_mind_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,

    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
#should add fp16=True for GPU training

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
'''

from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
audio_file = dataset[0]["audio"]["path"]


from transformers import pipeline, AutoTokenizer, Wav2Vec2FeatureExtractor

tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)


print(transcriber(audio_file))