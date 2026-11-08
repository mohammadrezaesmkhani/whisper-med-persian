"""
Surgical-Whisper — Stage 1 (Exp 101: Sergical-01) & Weight Merging
------------------------------------------------------------------
Surgically applies QLoRA (r=32, alpha=64) to layers 9..23 (skipping acoustic layers 0..8),
resulting in 21,626,880 trainable parameters on openai/whisper-medium.
After convergence, fuses the LoRA adapters into the base backbone for Stage 2.
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
    PeftModel,
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

            sentence = feature.get("sentence") or feature.get("text")
            if sentence is None:
                raise ValueError("Feature missing 'sentence' or 'text' key.")
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
    base_model_id = "openai/whisper-medium"
    dataset_path = "/data/whisper/dataset/CV-17-01/validated"
    stage1_output_dir = "/data/whisper/train/medium/101-sergical"
    merged_output_dir = "/data/whisper/train/medium/101-sergical/merged_model"

    validated_split = Dataset.load_from_disk(dataset_path)
    split_ds = validated_split.train_test_split(test_size=0.2, seed=42, shuffle=True)
    dataset_dict = DatasetDict({"train": split_ds["train"], "validation": split_ds["test"]})
    eval_dataset_small = dataset_dict["validation"].shuffle(seed=42).select(range(300))

    processor = AutoProcessor.from_pretrained(base_model_id, language="fa", task="transcribe")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    stage1_lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        layers_to_transform=list(range(9, 24)),
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, stage1_lora_config)
    model.__class__ = WhisperPeftModel
    model.print_trainable_parameters()

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_norm = [normalize_persian_text(t) for t in pred_str]
        label_norm = [normalize_persian_text(t) for t in label_str]
        wer = wer_metric.compute(predictions=pred_norm, references=label_norm)
        return {"wer": wer}

    total_samples = len(dataset_dict["train"])
    batch_size = 32
    grad_accum = 2
    steps_per_epoch = math.ceil(total_samples / (batch_size * grad_accum))
    total_epochs = 5
    eval_steps = int(steps_per_epoch * 0.2)

    training_args = Seq2SeqTrainingArguments(
        output_dir=stage1_output_dir,
        report_to="none",
        lr_scheduler_type="cosine",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-4,
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
        save_total_limit=10,
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

    adapter_save_path = f"{stage1_output_dir}/final_adapter"
    model.save_pretrained(adapter_save_path)
    processor.save_pretrained(adapter_save_path)

    print("Merging Stage 1 surgical adapters into base model...")
    base_model_fp16 = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base_model_fp16, adapter_save_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(merged_output_dir)
    processor.save_pretrained(merged_output_dir)
    print(f"Merged model successfully saved to: {merged_output_dir}")


if __name__ == "__main__":
    main()
