"""
Surgical-Whisper — Stage 2 (Exp 102: sergical-02) Continued Full-Layer QLoRA
----------------------------------------------------------------------------
Loads the merged Stage 1 model in 4-bit NormalFloat (NF4) and applies a fresh
LoRA adapter (r=32, alpha=64) across all 24 encoder/decoder layers (0..23),
yielding 34,603,008 trainable parameters (4.3337% of total parameters).
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
    PeftModelForSeq2SeqLM,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


@dataclass
class UnifiedDataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __post_init__(self):
        if hasattr(self.processor.tokenizer, "bos_token_id"):
            self.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        elif hasattr(self.processor.tokenizer, "decoder_start_token_id"):
            self.decoder_start_token_id = self.processor.tokenizer.decoder_start_token_id
        else:
            raise ValueError("Tokenizer must define 'bos_token_id' or 'decoder_start_token_id'.")

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features_list = []
        label_features_list = []

        for feature in features:
            audio = feature.get("audio")
            if audio is None:
                raise ValueError("Feature missing 'audio' key.")

            input_features = self.processor.feature_extractor(
                audio["array"],
                sampling_rate=audio.get("sampling_rate", 16000),
            ).input_features[0]
            input_features_list.append({"input_features": torch.tensor(input_features)})

            sentence = feature.get("sentence")
            if sentence is None:
                raise ValueError("Feature missing 'sentence' key.")
            labels = self.processor.tokenizer(sentence).input_ids
            label_features_list.append({"input_ids": labels})

        batch = self.processor.feature_extractor.pad(input_features_list, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features_list, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class WhisperPeftModel(PeftModelForSeq2SeqLM):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        input_features = kwargs.pop("input_features", None)
        forward_kwargs = {}
        if input_features is not None:
            forward_kwargs["input_features"] = input_features
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask
        if inputs_embeds is not None:
            forward_kwargs["inputs_embeds"] = inputs_embeds
        if decoder_input_ids is not None:
            forward_kwargs["decoder_input_ids"] = decoder_input_ids
        if decoder_attention_mask is not None:
            forward_kwargs["decoder_attention_mask"] = decoder_attention_mask
        if decoder_inputs_embeds is not None:
            forward_kwargs["decoder_inputs_embeds"] = decoder_inputs_embeds
        if labels is not None:
            forward_kwargs["labels"] = labels
        if output_attentions is not None:
            forward_kwargs["output_attentions"] = output_attentions
        if output_hidden_states is not None:
            forward_kwargs["output_hidden_states"] = output_hidden_states
        if return_dict is not None:
            forward_kwargs["return_dict"] = return_dict

        forward_kwargs.update(kwargs)
        return self.base_model(**forward_kwargs)


def normalize_persian_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = (
        text.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("ء", "")
        .replace("ؤ", "و")
        .replace("ئ", "ی")
        .replace("إ", "ا")
    )
    return re.sub(r"\s+", " ", text).strip()


def main():
    merged_model_path = "/data/whisper/train/medium/101-sergical/merged_model"
    dataset_path = "/data/whisper/dataset/CV-17-01/validated"
    output_dir = "/data/whisper/train/medium/102-sergical/continued_training"

    validated_split = Dataset.load_from_disk(dataset_path)
    split_ds = validated_split.train_test_split(test_size=0.2, seed=42, shuffle=True)
    dataset_dict = DatasetDict({"train": split_ds["train"], "validation": split_ds["test"]})
    eval_dataset_small = dataset_dict["validation"].shuffle(seed=42).select(range(300))

    processor = AutoProcessor.from_pretrained(merged_model_path)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        merged_model_path,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, config)
    model.__class__ = WhisperPeftModel
    model.print_trainable_parameters()

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str_normalized = [normalize_persian_text(text) for text in pred_str]
        label_str_normalized = [normalize_persian_text(text) for text in label_str]

        if not pred_str_normalized or not label_str_normalized:
            return {"wer": 1.0}

        wer = wer_metric.compute(predictions=pred_str_normalized, references=label_str_normalized)
        print(f"\n🎯 WER Score: {wer:.4f}")
        return {"wer": wer}

    total_samples = len(dataset_dict["train"])
    batch_size = 32
    gradient_accumulation_steps = 2
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(total_samples / effective_batch_size)
    total_epochs = 6
    eval_steps = int(steps_per_epoch * 0.2)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        report_to="none",
        lr_scheduler_type="cosine",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=5e-4,
        warmup_steps=int(0.1 * steps_per_epoch * total_epochs),
        num_train_epochs=total_epochs,
        eval_strategy="steps",
        eval_steps=eval_steps,
        fp16=False,
        bf16=True,
        per_device_eval_batch_size=batch_size,
        optim="adamw_torch",
        logging_steps=50,
        remove_unused_columns=False,
        label_names=["labels"],
        save_steps=eval_steps,
        save_total_limit=30,
        gradient_checkpointing=False,
        torch_compile=False,
        dataloader_pin_memory=False,
        predict_with_generate=True,
        generation_max_length=448,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=eval_dataset_small,
        data_collator=UnifiedDataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":
    main()
