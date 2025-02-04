import transformers
import datasets
from datasets import load_dataset
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import evaluate
import nltk
import numpy as np
import torch.nn.functional as F

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AdamW,
    set_seed,
)


def prepare_log_dir(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    exp_dir_folder_ls = os.listdir(exp_dir)
    if not exp_dir_folder_ls:
        exp_log_dir = os.path.join(exp_dir, f"{0}")
        os.makedirs(exp_log_dir)
    else:
        ls = []
        for i in range(len(exp_dir_folder_ls)):
            try:
                ls.append(int(exp_dir_folder_ls[i]))
            except:
                continue
        exp_dir_folder_ls = ls
        exp_dir_folder_ls.sort()
        exp_log_dir = os.path.join(exp_dir, f"{int(exp_dir_folder_ls[-1]) + 1}")
        os.makedirs(exp_log_dir)

    config_file_path = args.config
    shutil.copy(config_file_path, os.path.join(exp_log_dir, "config_T5.yml"))
    return exp_log_dir


def init_experiment(args, config, exp_type="train"):
    exp_log_dir = prepare_log_dir(args)
    args.output_dir = exp_log_dir

    for arg, value in vars(args).items():
        setattr(config, arg, value)

    if exp_type == "train":
        print(f"Saving log files to dir: {config.output_dir}")

    print("\n=========================================")
    print("Experiment Settings:")
    string = ""
    for arg, value in vars(config).items():
        string += f"({arg}: {value}) ; "
    print(string[0:-2])
    print("=========================================\n")
    return config


def load_model_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_data(dataset_name="achrafothman/aslg_pc12"):
    data = load_dataset(dataset_name)["train"]

    train_test_split = data.train_test_split(
        test_size=0.1, seed=40
    )  # 90% train, 10% test
    test_data = train_test_split["test"]
    train_data = train_test_split["train"]

    train_val_split = train_data.train_test_split(
        test_size=0.1, seed=40
    )  # 10% of train for validation
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]

    print("Train columns:", train_data.column_names)
    print("Total length of train data:", len(train_data))
    print("Total length of validation data:", len(val_data))
    print("Total length of test data:", len(test_data))

    print("\nSample data from train:")
    for i in range(3):
        print("Gloss: " + train_data["gloss"][i] + "Text: " + train_data["text"][i])

    return train_data, val_data, test_data


def preprocess_examples(batch, tokenizer, max_source_length, max_target_length):
    source = ["translate ASL to English: " + gloss for gloss in batch["gloss"]]
    target = batch["text"]

    # Tokenize the gloss (source)
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )

    # Tokenize the text (target)
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    # Prepare batch dictionary
    batch = {k: v for k, v in source_tokenized.items()}

    # Replace padding token IDs with -100 in the labels
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]

    return batch


def get_dataloader(data, batch_size):
    dataloader = DataLoader(data, shuffle=False, batch_size=batch_size)
    return dataloader


def save_training_state(
    base_path,
    optimizer,
    current_epoch,
    global_step,
    best_val_loss,
):
    print("Saving optimizer state")
    torch.save(optimizer.state_dict(), os.path.join(base_path, "optimizer.pt"))

    print("Saving RNG state...")
    torch.save(torch.get_rng_state(), os.path.join(base_path, "rng_state.pth"))

    print("Saving trainer state...")
    trainer_state = {
        "epoch": current_epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    torch.save(trainer_state, os.path.join(base_path, "trainer_state.json"))
    print(f"All training state files saved to {base_path}")


def compute_metrics(preds, labels, tokenizer):
    metric_rogue = evaluate.load("rouge")
    metric_bleu = evaluate.load("bleu")

    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric_rogue.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)

    # Compute BLEU
    preds_tokens = [pred.split() for pred in decoded_preds]
    labels_tokens = [[label.split()] for label in decoded_labels]
    preds_sentences = [" ".join(tokens) for tokens in preds_tokens]
    references_sentences = [[" ".join(ref) for ref in refs] for refs in labels_tokens]
    bleu_results = metric_bleu.compute(
        predictions=preds_sentences, references=references_sentences
    )["bleu"]
    result["bleu"] = bleu_results

    result = {k: round(v, 4) for k, v in result.items()}
    return result


def train_model(model, tokenizer, train_dataloader, val_dataloader, config):
    # Initialize accelerator
    accelerator = Accelerator()

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(config.seed)

    # Instantiate optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # Now we train the model
    epochs_no_improve = 0
    min_val_loss = float("inf")

    writer = SummaryWriter(log_dir=config.output_dir)

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            range(len(train_dataloader)), disable=not accelerator.is_main_process
        )
        progress_bar.set_description(f"Epoch: {epoch}")
        model.train()
        for i, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)
            global_step = (
                epoch * len(train_dataloader) + progress_bar.n
            )  # Calculate global step
            if i % config.logging_steps == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)

        model.eval()
        validation_losses = []
        cumulative_results = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "rougeLsum": 0.0,
            "bleu": 0.0,
            "gen_len": 0.0,
        }
        for i, batch in tqdm(enumerate(val_dataloader), desc="Running Validation"):
            with torch.no_grad():
                # loss
                outputs = model(**batch)
                # rouge values
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                gen_outputs = model.generate(
                    input_ids, attention_mask=attention_mask
                ).to("cpu")
                labels = batch["labels"].to("cpu")
                results = compute_metrics(gen_outputs, labels, tokenizer)

            loss = outputs.loss
            validation_losses.append(accelerator.gather(loss[None]))
            for key in cumulative_results.keys():
                cumulative_results[key] += results[key]

        val_loss = torch.stack(validation_losses).sum().item() / len(validation_losses)
        writer.add_scalar("eval/loss", val_loss, global_step)

        cumulative_results = {
            key: value / len(validation_losses)
            for key, value in cumulative_results.items()
        }
        for key, val in cumulative_results.items():
            writer.add_scalar(f"eval/{key}", val, global_step)

        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: validation loss:", val_loss)
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            # Save model when validation loss improves
            pth = os.path.join(config.output_dir, "best_model")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(pth, save_function=accelerator.save)
            tokenizer.save_pretrained(pth)
            save_training_state(
                base_path=pth,
                optimizer=optimizer,
                current_epoch=epoch,
                global_step=global_step,
                best_val_loss=min_val_loss,
            )
            continue
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == config.patience:
                accelerator.print("Early stopping!")
                break

    # save trained model
    accelerator.wait_for_everyone()
    pth = os.path.join(config.output_dir, "final_model")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(pth, save_function=accelerator.save)
    tokenizer.save_pretrained(pth)
    save_training_state(
        base_path=pth,
        optimizer=optimizer,
        current_epoch=epoch,
        global_step=global_step,
        best_val_loss=min_val_loss,
    )
    writer.close()


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def calculate_perplexity(model, tokenizer, input_ids, attention_mask, batch):
    batch = {
        key: val.to(model.device)
        for key, val in batch.items()
        if isinstance(val, torch.Tensor)
    }
    logits = model(**batch).logits
    shift_logits = logits[:, :-1, :].contiguous()

    labels = batch["labels"].to("cpu")
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    shift_labels = torch.from_numpy(labels[:, 1:]).to(model.device).contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, reduction="sum"
    )
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss / shift_labels.numel())
    return perplexity.detach().cpu().numpy()


def generate_rich_text(
    test_dataloader, model, tokenizer, encoder_max_length, compute_metrics=True
):
    model.eval()
    predictions = []
    references = []
    glosses = []
    outputs_list = []
    perplexities = []

    for batch in tqdm(test_dataloader, desc="Evaluating on Test Set"):
        with torch.no_grad():
            # Prepare inputs for the model
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            # Generate predictions for the batch
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=encoder_max_length,
                num_beams=4,  # Beam search
            )

            # Decode predictions and references
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels = batch["labels"].to("cpu")
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            gloss_texts = batch["input_ids"].to("cpu")
            gloss_texts = np.where(
                gloss_texts != -100, gloss_texts, tokenizer.pad_token_id
            )
            gloss_texts = tokenizer.batch_decode(gloss_texts, skip_special_tokens=True)
            gloss_texts = [
                txt.replace("translate ASL to English: ", "") for txt in gloss_texts
            ]

            # Store outputs and raw text
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)
            glosses.extend(gloss_texts)
            outputs_list.extend(outputs.cpu().tolist())
            perplexities.append(
                calculate_perplexity(model, tokenizer, input_ids, attention_mask, batch)
            )

    # Extract 10 long sentences for display
    long_examples = []
    for gloss, pred, ref in zip(glosses, predictions, references):
        if len(pred.split()) > 15:  # Filter for long sentences (longer than 15 words)
            long_examples.append(
                {"gloss": gloss, "ground_truth": ref, "prediction": pred}
            )
        if len(long_examples) >= 10:  # Collect only 10 examples
            break

    results = None
    if compute_metrics:
        preds, labels = postprocess_text(predictions, references)

        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")

        # Compute ROUGE
        rouge_results = rouge.compute(
            predictions=preds, references=labels, use_stemmer=True
        )

        # Compute BLEU
        preds_tokens = [pred.split() for pred in preds]
        labels_tokens = [[label.split()] for label in labels]
        preds_sentences = [" ".join(tokens) for tokens in preds_tokens]
        references_sentences = [
            [" ".join(ref) for ref in refs] for refs in labels_tokens
        ]

        bleu_results = bleu.compute(
            predictions=preds_sentences, references=references_sentences
        )
        results = {"rouge": rouge_results, "bleu": bleu_results}

    perplexity = sum(perplexities) / len(perplexities)
    results["perplexity"] = perplexity
    return outputs_list, predictions, results, long_examples
