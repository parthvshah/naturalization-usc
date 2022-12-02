import json
import random
from collections import OrderedDict

import nltk
import numpy as np

from model.utils import (
    END_TOKEN,
    START_TOKEN,
    load_corpus,
    load_disfluencies,
    save_model,
    load_model,
)


def h_inc_dict(dict, k):
    if k in dict.keys():
        dict[k] = dict[k] + 1
    else:
        dict[k] = 1


def learn_bigrams(raw_corpus, flat_disfluencies, proportionalize=True):
    ct_total_bigrams = 0
    ct_total_nonbigrams = 0

    all_bigrams = {}
    non_bigrams = {}
    pos_bigrams = {}

    for line in raw_corpus:
        tokens = line.split()
        tokens = [START_TOKEN] + tokens + [END_TOKEN]
        # TODO: add line start line end tokens

        pos_tags = nltk.pos_tag(tokens)

        for i in range(len(tokens) - 1):
            first_w = tokens[i]
            second_w = tokens[i + 1]
            first_w_pos = pos_tags[i]
            second_w_pos = pos_tags[i + 1]

            bigram = tuple((first_w, second_w))
            if (
                first_w in flat_disfluencies or second_w in flat_disfluencies
            ) and not ():
                ct_total_bigrams = ct_total_bigrams + 1
                h_inc_dict(all_bigrams, bigram)

                if first_w in flat_disfluencies:
                    if second_w == START_TOKEN:  # Should never be the case
                        pos_bigram_tags = tuple((first_w, START_TOKEN))
                    elif second_w == END_TOKEN:
                        pos_bigram_tags = tuple((first_w, END_TOKEN))
                    else:
                        pos_bigram_tags = tuple((first_w, second_w_pos[-1]))
                elif second_w in flat_disfluencies:
                    if first_w == START_TOKEN:
                        pos_bigram_tags = tuple((START_TOKEN, second_w))
                    elif first_w == END_TOKEN:  # Should never be the case
                        pos_bigram_tags = tuple((END_TOKEN, second_w))
                    else:
                        pos_bigram_tags = tuple((first_w_pos[-1], second_w))

                h_inc_dict(pos_bigrams, pos_bigram_tags)
            else:
                pass
                # if first_w[0] == "(" and first_w[-1] == ")":
                #     print(#raise Exception(
                #         f"Error! Found a bigram in the training corpus that was not acceptable. Error Bigram: {first_w}"
                #     )
                # elif second_w[0] == "(" and second_w[-1] == ")":
                #     print(#raise Exception(
                #         f"Error! Found a bigram in the training corpus that was not acceptable. Error Bigram: {second_w}"
                #     )

                ct_total_nonbigrams = ct_total_nonbigrams + 1
                h_inc_dict(non_bigrams, bigram)

    if proportionalize:
        for k in all_bigrams.keys():
            all_bigrams[k] = all_bigrams[k] / ct_total_bigrams
        for j in pos_bigrams.keys():
            pos_bigrams[j] = pos_bigrams[j] / ct_total_bigrams

    return ct_total_bigrams, ct_total_nonbigrams, all_bigrams, non_bigrams, pos_bigrams


def index_bigrams(all_bigrams):
    idx_by_first = {}
    idx_by_second = {}

    for bigram, val in all_bigrams.items():
        first_w = bigram[0]
        second_w = bigram[1]

        if first_w not in idx_by_first.keys():
            idx_by_first[first_w] = {}
        idx_by_first[first_w][bigram] = val

        if second_w not in idx_by_second.keys():
            idx_by_second[second_w] = {}
        idx_by_second[second_w][bigram] = val

    return idx_by_first, idx_by_second


def apply_disfluencies(tokens, d_locations, d_choices):
    d_locations = d_locations[::-1]
    d_choices = d_choices[::-1]

    for i, loc in enumerate(d_locations):
        tokens = tokens[: loc + 1] + [d_choices[i]] + tokens[loc + 1 :]

    return " ".join(tokens[1:-1])


def gen_bigrams(
    selection_percent,
    all_bigrams,
    test_corpus_path,
    pos_bigrams,
    output_path=None,
    token_strategy="pos_selection_next",
    disf_strategy="random",
    test_in_sen=None,
):
    if test_corpus_path is None:
        test_corpus = [test_in_sen]
    else:
        test_corpus = load_corpus(test_corpus_path)

    idx_by_first, idx_by_second = index_bigrams(all_bigrams)
    pos_by_first, pos_by_second = index_bigrams(pos_bigrams)

    if output_path is not None:
        write_file = open(output_path, "w+")

    for line in test_corpus:
        tokens = line.split()
        tokens = [START_TOKEN] + tokens + [END_TOKEN]
        d_locations = []
        d_choices = []
        pos_tags = nltk.pos_tag(tokens)

        # How we choose tokens
        if token_strategy == "random_between_possible":
            actionable_idxs = [
                x
                if (
                    tokens[x] in idx_by_first.keys()
                    or tokens[x + 1] in idx_by_second.keys()
                )
                else None
                for x in list(range(len(tokens) - 1))
            ]
            actionable_idxs = [x for x in actionable_idxs if x is not None]
            num_items_to_alter = int((selection_percent * len(actionable_idxs)) // 1)
            indexes_to_update = random.sample(actionable_idxs, num_items_to_alter)
        elif token_strategy == "random_between_global":
            num_items_to_alter = int((selection_percent * len(tokens)) // 1)
            actionable_idxs = [
                x
                if (
                    tokens[x] in idx_by_first.keys()
                    or tokens[x + 1] in idx_by_second.keys()
                )
                else None
                for x in list(range(len(tokens) - 1))
            ]
            actionable_idxs = [x for x in actionable_idxs if x is not None]
            num_items_to_alter = min(num_items_to_alter, len(actionable_idxs))
            indexes_to_update = random.sample(actionable_idxs, num_items_to_alter)
        elif token_strategy == "pos_selection_next":
            actionable_idxs = []
            for x in list(range(len(tokens) - 1)):
                pos_f = pos_tags[x][0]
                tag_f = pos_tags[x][-1]

                f_prob = 0

                if pos_f == START_TOKEN:
                    f_prob = sum(pos_by_first[pos_f].values()) / 10
                elif tag_f in pos_by_first.keys():
                    f_prob = sum(pos_by_first[tag_f].values())

                actionable_idxs.append(f_prob)

            idxs = list(range(len(actionable_idxs)))
            num_items_to_alter = int((selection_percent * len(actionable_idxs)) // 1)
            indexes_to_update = random.choices(
                idxs, actionable_idxs, k=num_items_to_alter
            )
            indexes_to_update = list(set(indexes_to_update))
        elif token_strategy == "pos_selection_next_and_prev":

            actionable_idxs = []
            # for x in list(range(len(tokens) - 1)):
            #     pos_f = pos_tags[x][0]
            #     tag_f = pos_tags[x][-1]
            #     pos_s = pos_tags[x+1][0]
            #     tag_s = pos_tags[x+1][-1]
            #     first_prob = 0
            #     second_prob = 0
            #     if pos_f == START_TOKEN:
            #         first_prob = pos_by_first[pos_f]
            #     elif tag_f in pos_by_second.keys():
            #     if pos_s == END_TOKEN:
            #         second_prob = pos_by_second[pos_s]
            #     else:
            #         actionable_idxs.append[pos_by_second[END_TOKEN]]
            # actionable_idxs = [x for x in actionable_idxs if x is not None]
            # num_items_to_alter = int((selection_percent * len(actionable_idxs)) // 1)
            # indexes_to_update = random.sample(actionable_idxs, num_items_to_alter)
        else:
            raise NotImplementedError(
                f"Token selection strategy '{token_strategy}' is not implemented!"
            )

        indexes_to_update.sort()

        # How we choose disfluencies for the selected tokens
        for idx in indexes_to_update:
            first_w = tokens[idx]
            first_w_pos = pos_tags[idx][-1]
            if idx + 1 < len(tokens):
                second_w = tokens[idx + 1]
                second_w_pos = pos_tags[idx + 1][-1]

            if disf_strategy == "random":
                possible_dis = OrderedDict()
                if first_w in idx_by_first.keys():
                    temp = idx_by_first[first_w]
                    # for k,v in temp.items()
                    possible_dis.update(temp)
                if idx + 1 < len(tokens) and second_w in idx_by_second.keys():
                    temp = idx_by_second[second_w]
                    possible_dis.update(temp)

                norm_vals = [
                    x / sum(list(possible_dis.values()))
                    for x in list(possible_dis.values())
                ]
                if len(norm_vals) > 0:
                    d_choice = list(possible_dis.keys())[
                        np.random.choice(
                            list(range(len(possible_dis.keys()))), p=norm_vals
                        )
                    ]

                    if (
                        d_choice[-1][0] == "(" and d_choice[-1][-1] == ")"
                    ):  # TODO: replace with item matching instead of string checking
                        d_choice = d_choice[-1]
                    else:
                        d_choice = d_choice[0]
                    d_locations.append(idx)
                    d_choices.append(d_choice)

            else:
                raise NotImplementedError(
                    f"Disfluency selection strategy '{disf_strategy}' is not implemented!"
                )

        out_string = apply_disfluencies(tokens, d_locations, d_choices)
        if output_path is not None:
            write_file.write(out_string + "\n")

    if output_path is None:
        return out_string
    else:
        write_file.close()


def process_bigrams():
    flat_disfluencies = load_disfluencies(
        d_path="../data/santa_barabara_data/disfluency_key.json",
        acceptable_dis=["Pause", "Extra"],
    )
    proc_corpus = load_corpus(
        c_path="../data/santa_barabara_data/sb_full_insertions_transcription.txt",
        clean_filters=None,
    )
    (
        ct_total_bigrams,
        ct_total_nonbigrams,
        all_bigrams,
        non_bigrams,
        pos_bigrams,
    ) = learn_bigrams(proc_corpus, flat_disfluencies)

    save_model(model=all_bigrams, path="./bigram_model.pkl", seq_len=pos_bigrams)

    test_in_path = "../data/test_input.txt"
    output_path = "../data/test_output_0.35_scaled_start.txt"
    selection_percent = 0.35
    gen_bigrams(
        selection_percent,
        all_bigrams,
        test_in_path,
        pos_bigrams,
        output_path=output_path,
    )


def map_to_speechtext(sentence, disfluencies):
    invert_disfluencies = {v: k for k, v in disfluencies.items()}
    out_words = []
    for word in sentence.split():
        if word in invert_disfluencies.keys():
            out_words.append(invert_disfluencies[word])
        else:
            out_words.append(word)
    final_string = " ".join(out_words)
    return final_string


def online_process(
    input_sentence, selection_percent=0.3, model_path="./bigram_model.pkl"
):
    bigram_model = load_model(path=model_path)
    bg_string = gen_bigrams(
        selection_percent,
        bigram_model,
        test_corpus_path=None,
        output_path=None,
        test_in_sen=input_sentence,
    )
    disfluencies = load_disfluencies(
        d_path="../data/santa_barabara_data/disfluency_key.json",
        acceptable_dis=["Pause", "Extra"],
        return_dict=True,
    )
    mapped_string = map_to_speechtext(sentence=bg_string, disfluencies=disfluencies)
    return mapped_string


if __name__ == "__main__":
    process_bigrams()
    online_test = online_process("Look I think there's a dog over there")
    print(online_test)
