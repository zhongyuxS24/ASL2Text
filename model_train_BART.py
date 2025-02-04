import transformers
from datasets import load_dataset
import yaml
import argparse

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from utils.utils_BART import *
import nltk

nltk.download("punkt", quiet=True)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args():
    parser = argparse.ArgumentParser(description="BART train")
    parser.add_argument(
        "--exp_name",
        default="BART_train",
        type=str,
        help="experiment name",
    )
    parser.add_argument(
        "--config",
        default="config/config_BART.yml",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="output path to save weights and tensorboard logs",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="resume training from checkpoint",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    return args, config


def main():
    args, config = parse_args()
    config = init_experiment(args, config)

    if config.resume:
        config.model_name = config.resume

    model, tokenizer = load_model_tokenizer(model_name=config.model_name)
    train_data, val_data, _ = load_data(dataset_name=config.dataset_name)

    # Tokenize and preprocess data
    train_data = train_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, config.max_source_length, config.max_target_length
        ),
        batched=True,
        remove_columns=train_data.column_names,
    )

    val_data = val_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, config.max_source_length, config.max_target_length
        ),
        batched=True,
        remove_columns=val_data.column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        do_train=config.do_train,
        do_eval=config.do_eval,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        label_smoothing_factor=config.label_smoothing_factor,
        predict_with_generate=config.predict_with_generate,
        logging_dir=config.output_dir,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        eval_strategy=config.evaluation_strategy,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    trainer.train()


if __name__ == "__main__":
    main()
