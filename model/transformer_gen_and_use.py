import json
import os
import random
from collections import OrderedDict

import nltk as nltk
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AdamW

from eval.evaluation import similar_sentence_score, similar_insertion_score
from utils import (
    load_disfluencies,
    load_corpus,
    START_TOKEN,
    END_TOKEN,
    save_model,
    load_model,
    PAD_TOKEN,
    filter_raw,
)

EPOCHS = 100
LR = 5e-3
BATCH_SIZE = 16


def tokenize_and_pad_corpus(tokenizer, corpus, device, seq_len=None):
    all_ids = []
    for i in range(len(corpus)):
        corpus[i] = START_TOKEN + " " + corpus[i] + " " + END_TOKEN

    if seq_len is None:
        tokenized = tokenizer(corpus, return_tensors="pt", padding="longest")
        seq_len = tokenized.input_ids.shape[1]
    else:
        tokenizer.model_max_length = seq_len
        tokenized = tokenizer(corpus, return_tensors="pt", padding="max_length")
    tok_sen = tokenized.input_ids
    input_ids = tok_sen.clone().to(device)
    input_ids.to(torch.long)

    return input_ids, seq_len


def tokenize_and_pad_sen(tokenizer, sentence, device, seq_len):
    tokenizer.model_max_length = seq_len
    tokenized = tokenizer(sentence, return_tensors="pt", padding="max_length")

    tok_sen = tokenized.input_ids
    input_ids = tok_sen.to(device)

    return input_ids.to(torch.long)


def evaluate(model, tokenizer, device, seq_len, flat_disfluencies):
    test_file_unfiltered = "data/santa_barabara_data/sb_full_test_raw.txt"
    test_file_in = "data/santa_barabara_data/sb_full_test_filtered.txt"
    test_file_predictions = "data/santa_barabara_data/tmp_test_predictions.txt"

    if os.path.exists(test_file_predictions):
        os.remove(test_file_predictions)

    filter_raw(test_file_unfiltered, test_file_in)

    autoregressive_predict(
        model,
        tokenizer,
        test_file_in,
        test_file_predictions,
        device,
        seq_len,
        flat_disfluencies,
        autoreg=False,
    )

    nltk.download("averaged_perceptron_tagger")
    sis = similar_insertion_score(test_file_predictions, test_file_unfiltered)

    print("Similar insertion score on corpus:", sis)


def learn_transformer(
    in_corpus, lbl_corpus, model, tokenizer, device, flat_disfluencies
):
    optimizer = AdamW(model.parameters(), lr=LR)

    labels_corpus, seq_len = tokenize_and_pad_corpus(tokenizer, lbl_corpus, device)
    input_corpus, _ = tokenize_and_pad_corpus(tokenizer, in_corpus, device, seq_len)
    loss_fn = CrossEntropyLoss()

    for epoch in range(EPOCHS):
        progress_bar = tqdm(
            range(BATCH_SIZE, len(input_corpus), BATCH_SIZE), position=0, leave=True
        )
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = 0

        data_idx = 0
        for ids_idx in range(BATCH_SIZE, len(input_corpus), BATCH_SIZE):
            input_ids = input_corpus[ids_idx : ids_idx + BATCH_SIZE]
            labels = labels_corpus[ids_idx : ids_idx + BATCH_SIZE]

            pos_tags = nltk.pos_tag(in_corpus[data_idx])

            # Exec
            mask = gen_mask(
                input_ids, input_ids.shape[-1], device, tokenizer.pad_token_id
            )
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=mask)
            loss = outputs.loss
            # loss = loss_fn(outputs.logits.transpose(-1,-2), labels)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            # print(f"Loss: {loss}")
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            epoch_loss += loss.item()
            data_idx += 1

        evaluate(model, tokenizer, device, seq_len, flat_disfluencies)
        print(
            f"\nEpoch {epoch} average loss is: {epoch_loss/((len(input_corpus) / BATCH_SIZE))}"
        )
        # progress_bar.display(msg=f"\tEpoch {epoch} average loss is: {epoch_loss/ ((len(input_corpus) / BATCH_SIZE))}")

        if (epoch + 1) % 10 == 0 or epoch == EPOCHS:
            save_model(model, seq_len, path=f"model_epoch_{epoch}.pt")

    return model, seq_len


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
    # tokenizer.model_max_length = STRIDE
    special_tokens_dict = {
        "bos_token": START_TOKEN,
        "eos_token": END_TOKEN,
        "pad_token": PAD_TOKEN,
    }
    tokenizer.add_tokens(new_tokens=flat_disfluencies)
    tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))
    return tokenizer


def gen_mask(tokens, start_loc, device, pad_token_id):
    padded = []
    for sentence in tokens:
        padded.append(
            (torch.tensor([1 if x != pad_token_id else 0 for x in sentence])).to(device)
        )
    return torch.stack(padded, dim=0)


def autoregressive_predict(
    model,
    tokenizer,
    test_in_path,
    output_path,
    device,
    seq_len,
    disfluencies,
    autoreg=True,
):
    test_corpus = load_corpus(test_in_path)
    write_file = open(output_path, "w+")

    # test_corpus, _ = tokenize_and_pad_corpus(tokenizer, test_raw, device, seq_len)
    for line in test_corpus:
        split_line = [START_TOKEN] + line.split() + [END_TOKEN]
        tokens = tokenize_and_pad_sen(tokenizer, " ".join(split_line), device, seq_len)
        working_sentence = []

        if autoreg:
            for pos in range(len(split_line) - 1):
                last_word_so_far = split_line[pos]
                mask = gen_mask(tokens, pos, device, tokenizer.pad_token_id)

                with torch.no_grad():
                    outputs = model(
                        input_ids=tokens, labels=tokens, attention_mask=mask
                    )

                    # TODO: filter for only probs of disfluency tokens
                    out_words = torch.argmax(outputs.logits, dim=-1).tolist()[0]
                    next_selected_token = tokenizer.decode(token_ids=out_words[pos + 1])
                    if last_word_so_far not in tokenizer.all_special_tokens:
                        working_sentence.append(last_word_so_far)
                    if next_selected_token in disfluencies:
                        working_sentence.append(next_selected_token)
            working_sentence.append("\n")
            working_sentence = " ".join(working_sentence)
        else:
            with torch.no_grad():
                mask = (
                    (
                        torch.tensor(
                            [
                                1 if x != tokenizer.pad_token_id else 0
                                for x in tokens.tolist()[0]
                            ]
                        )
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                outputs = model(input_ids=tokens, labels=tokens, attention_mask=mask)

                # TODO: filter for only probs of disfluency tokens
                out_words = torch.argmax(outputs.logits, dim=-1).tolist()[0]
                out_words = [w for w in out_words if w != tokenizer.pad_token_id]
                words = tokenizer.decode(token_ids=out_words)
                working_sentence = words + "\n"

        write_file.write(working_sentence)


def process_transformer(
    train=True, load_path=None, dis_path=None, in_txt_path=None, in_lbl_path=None
):
    flat_disfluencies = load_disfluencies(d_path=dis_path)
    in_corpus = load_corpus(c_path=in_txt_path)
    lbl_corpus = load_corpus(c_path=in_lbl_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_id = "gpt2"

    if train:
        model = get_model(model_id, device)
        tokenizer = get_tokenizer(model, flat_disfluencies, model_id)
        trained_model, seq_len = learn_transformer(
            in_corpus, lbl_corpus, model, tokenizer, device, flat_disfluencies
        )
    else:
        trained_model, seq_len = load_model(load_path, device)
        tokenizer = get_tokenizer(trained_model, flat_disfluencies, model_id)

    test_in_path = "./data/test_input.txt"
    output_path = "../data/test_output_29.txt"

    autoregressive_predict(
        trained_model,
        tokenizer,
        test_in_path,
        output_path,
        device,
        seq_len,
        flat_disfluencies,
    )


if __name__ == "__main__":
    dis_path = "data/santa_barabara_data/disfluency_key.json"
    full_select = True

    if full_select:
        in_raw_path = "data/santa_barabara_data/sb_full_insertions_transcription.txt"
        in_txt_path = "data/santa_barabara_data/filtered_full.txt"
    else:
        in_raw_path = "data/santa_barabara_data/sb_select_insertions_transcription.txt"
        in_txt_path = "data/santa_barabara_data/filtered_select.txt"

    in_lbl_path = in_raw_path

    if not os.path.exists(in_txt_path):
        filter_raw(in_raw_path, in_txt_path)

    process_transformer(
        dis_path=dis_path, in_txt_path=in_txt_path, in_lbl_path=in_lbl_path
    )
    # process_transformer(train=False, load_path='model_epoch_29.pt')
