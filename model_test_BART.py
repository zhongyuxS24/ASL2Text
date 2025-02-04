import transformers
from datasets import load_dataset
import torch
import yaml
import argparse
import json

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
        default="BART_test",
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
        default="inference_results",
        type=str,
        help="output path to save weights and tensorboard logs",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Weight path for running inference",
    )
    parser.add_argument(
        "--num_samples",
        default=5,
        type=int,
        help="num of samples to run inference on",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    return args, config


def main():
    args, config = parse_args()
    config = init_experiment(args, config)

    model, tokenizer = load_model_tokenizer(model_name=config.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    _, _, test_data = load_data(dataset_name=config.dataset_name)
    test_data = test_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, config.max_source_length, config.max_target_length
        ),
        batched=True,
        remove_columns=test_data.column_names,
    )
    test_data.set_format(type="torch")

    test_dataloader = get_dataloader(test_data, config.per_device_eval_batch_size)

    outputs, predictions, results, long_examples = generate_rich_text(
        test_dataloader,
        model,
        tokenizer,
        config.max_source_length,
        compute_metrics=True,
    )

    txt_file = os.path.join(config.output_dir, "results.txt")
    with open(txt_file, "w") as f:
        f.write("\n--- Test Set Metrics ---\n")
        f.write(json.dumps(results, indent=4))

        f.write("\n--- 10 Long Examples ---\n")
        for example in long_examples:
            f.write(f"Gloss: {example['gloss']}\n")
            f.write(f"Ground Truth: {example['ground_truth']}\n")
            f.write(f"Prediction: {example['prediction']}\n")
            f.write("===============================================\n")


if __name__ == "__main__":
    main()
