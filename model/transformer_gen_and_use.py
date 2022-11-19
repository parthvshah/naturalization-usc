import json
import random
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AdamW

from model.utils import (
    load_disfluencies,
    load_corpus,
    START_TOKEN,
    END_TOKEN,
    save_model,
    load_model,
    PAD_TOKEN,
)

STRIDE = 128
EPOCHS = 10
LR = 5e-3


def tokenize_and_pad(tokenizer, sentence, device):
    sentence = " ".join(sentence)
    tokenized = tokenizer(sentence, return_tensors="pt", padding="max_length")
    tok_sen = tokenized.input_ids
    seq_len = tok_sen.shape[-1]
    if seq_len > STRIDE:
        raise Exception(
            "Found a sentence that was longer than the expected transformer stride!"
        )
    input_ids = tok_sen.clone().to(device)

    return input_ids.to(torch.long)


def learn_trainsformer(proc_corpus, model, tokenizer, device):
    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        progress_bar = tqdm(range(len(proc_corpus)))
        progress_bar.set_description(f"Epoch {epoch}")

        epoch_loss = 0

        for sentence in proc_corpus:
            # Padding
            input_ids = tokenize_and_pad(tokenizer, sentence, device)

            # Exec
            batch = {"input_ids": input_ids, "labels": input_ids}
            model.resize_token_embeddings(len(tokenizer))
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            # print(f"Loss: {loss}")
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            epoch_loss += loss.item()
        # print(f"\nEpoch {epoch} average loss is: {epoch_loss/len(proc_corpus)}")
        progress_bar.display(
            msg=f"\nEpoch {epoch} average loss is: {epoch_loss/len(proc_corpus)}"
        )
        save_model(model, path=f"model_epoch_{epoch}.pt")

    return model


def get_model(model_id, device, pretrained=True):
    model = None
    if pretrained:
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    else:
        model

    if model is None:
        raise NotImplementedError("The model was not correctley implemented!")
    return model


def get_tokenizer(model, flat_disfluencies, model_id):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    tokenizer.model_max_length = STRIDE
    special_tokens_dict = {
        "bos_token": START_TOKEN,
        "eos_token": END_TOKEN,
        "pad_token": PAD_TOKEN,
    }
    tokenizer.add_tokens(new_tokens=flat_disfluencies)
    tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))
    return tokenizer


def autoregressive_predict(model, tokenizer, test_in_path, output_path, device):
    test_corpus = load_corpus(test_in_path)
    write_file = open(output_path, "w+")

    for line in test_corpus:
        working_sentence = ""
        split_line = [START_TOKEN] + line.split() + [END_TOKEN]

        for pos in range(len(split_line)):
            last_word_so_far = split_line[pos]
            tokens = tokenize_and_pad(tokenizer, split_line[: pos + 1], device)
            mask = torch.cat(
                (torch.ones((pos + 1)), torch.zeros((tokens.size(1) - (pos + 1)))),
                dim=0,
            ).to(device)

            with torch.no_grad():
                outputs = model(input_ids=tokens, labels=tokens, attention_mask=mask)

                # TODO: filter for only probs of disfluency tokens
                out_words = torch.argmax(outputs.logits, dim=-1).tolist()[0]
                out_str = tokenizer.decode(token_ids=out_words)
                next_selected_token = out_str.split()  # [pos+1]
                next_selected_token = "TEST"
                working_sentence = (
                    working_sentence
                    + " "
                    + last_word_so_far
                    + " "
                    + next_selected_token
                )


def process_transformer(train=True, load_path=None):
    flat_disfluencies = load_disfluencies()
    proc_corpus = load_corpus()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_id = "gpt2"

    if train:
        model = get_model(model_id, device)
        tokenizer = get_tokenizer(model, flat_disfluencies, model_id)
        trained_model = learn_trainsformer(proc_corpus, model, tokenizer, device)
    else:
        trained_model = load_model(load_path, device)
        tokenizer = get_tokenizer(trained_model, flat_disfluencies, model_id)

    test_in_path = "./data/test_input.txt"
    output_path = "./data/test_output.txt"

    autoregressive_predict(trained_model, tokenizer, test_in_path, output_path, device)


# process_transformer()
process_transformer(train=False, load_path="model_epoch_9.pt")
