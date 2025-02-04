import transformers
from datasets import load_dataset
import yaml

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from utils.utils_T5 import *
import nltk
import argparse

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
    parser = argparse.ArgumentParser(description="T5 train")
    parser.add_argument(
        "--exp_name",
        default="T5_train",
        type=str,
        help="experiment name",
    )
    parser.add_argument(
        "--config",
        default="config/config_T5.yml",
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
        lambda batch: preprocess_examples(
            batch, tokenizer, config.max_input_length, config.max_target_length
        ),
        batched=True,
        remove_columns=train_data.column_names,
    )
    train_data.set_format(type="torch")

    val_data = val_data.map(
        lambda batch: preprocess_examples(
            batch, tokenizer, config.max_input_length, config.max_target_length
        ),
        batched=True,
        remove_columns=val_data.column_names,
    )
    val_data.set_format(type="torch")

    train_dataloader = get_dataloader(train_data, config.train_batch_size)
    val_dataloader = get_dataloader(val_data, config.eval_batch_size)

    train_model(model, tokenizer, train_dataloader, val_dataloader, config)


if __name__ == "__main__":
    main()
