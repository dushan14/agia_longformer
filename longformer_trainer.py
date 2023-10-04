from datasets import load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

import data_loader


def process_data_to_model_inputs(batch):
    global tokenizer
    global encoder_max_length
    global decoder_max_length

    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["abstract"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


# compute Rouge score during validation
def compute_metrics(pred):
    global tokenizer
    global rouge

    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def process(args):
    global tokenizer
    global encoder_max_length
    global decoder_max_length
    global rouge

    rouge = load_metric("rouge")
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")


    # data
    train_data = data_loader.get_train_data(args)
    validation_data = data_loader.get_validation_data(args)

    # max encoder length is 8192 for PubMed
    encoder_max_length = 8192
    decoder_max_length = 512
    batch_size = args.batch_size

    # map train data
    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["article", "abstract", "section_names"],
    )

    # map val data
    validation_data = validation_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["article", "abstract", "section_names"],
    )

    # set Python list to PyTorch tensor
    train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    # set Python list to PyTorch tensor
    validation_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    # enable fp16 apex training
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        fp16_backend="apex.contrib",
        output_dir="./",
        logging_steps=250,
        eval_steps=5000,
        save_steps=500,
        warmup_steps=1500,
        save_total_limit=2,
        gradient_accumulation_steps=4,
    )

    # load model + enable gradient checkpointing & disable cache for checkpointing
    led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-large-16384", gradient_checkpointing=True, use_cache=False)

    # set generate hyperparameters
    led.config.num_beams = 4
    led.config.max_length = 512
    led.config.min_length = 100
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=led,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=validation_data,
    )

    # start training
    trainer.train()

    # saving model & tokenizer
    trainer.save_model(args.model_output_path)
    tokenizer.save_pretrained(args.tokenizer_output_path)
