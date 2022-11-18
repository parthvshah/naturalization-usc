import json
import random
from collections import OrderedDict
import numpy as np


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

    for line in raw_corpus:
        tokens = line.split()
        tokens = [START_TOKEN] + tokens + [END_TOKEN]
        #TODO: add line start line end tokens

        for i in range(len(tokens)-1):
            first_w = tokens[i]
            second_w = tokens[i+1]

            bigram = tuple((first_w, second_w))
            if first_w in flat_disfluencies or second_w in flat_disfluencies:
                ct_total_bigrams = ct_total_bigrams + 1
                h_inc_dict(all_bigrams, bigram)
            else:
                if (first_w[0] == "(" and first_w[-1] == ")"):
                    raise Exception(f"Error! Found a bigram in the training corpus that was not acceptable. Error Bigram: {first_w}")
                elif (second_w[0] == "(" and second_w[-1] == ")"):
                    raise Exception(f"Error! Found a bigram in the training corpus that was not acceptable. Error Bigram: {second_w}")

                ct_total_nonbigrams = ct_total_nonbigrams + 1
                h_inc_dict(non_bigrams, bigram)


    if proportionalize:
        for k in all_bigrams.keys():
            all_bigrams[k] = all_bigrams[k] / ct_total_bigrams

    return ct_total_bigrams, ct_total_nonbigrams, all_bigrams, non_bigrams


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
        tokens = tokens[:loc+1] + [d_choices[i]] + tokens[loc+1:]

    return " ".join(tokens[1:-1])


def gen_bigrams(selection_percent, all_bigrams, test_corpus_path, output_path, token_strategy="random_between_possible", disf_strategy="random"):
    test_corpus = load_corpus(test_corpus_path)
    idx_by_first, idx_by_second = index_bigrams(all_bigrams)

    write_file = open(output_path, "w+")

    for line in test_corpus:
        tokens = line.split()
        tokens = [START_TOKEN] + tokens + [END_TOKEN]
        d_locations = []
        d_choices = []

        #How we choose tokens
        if token_strategy == "random_between_possible":
            actionable_idxs = [x if (tokens[x] in idx_by_first.keys() or tokens[x+1] in idx_by_second.keys()) else None for x in list(range(len(tokens)-1))]
            actionable_idxs = [x for x in actionable_idxs if x is not None]
            num_items_to_alter = int((selection_percent * len(actionable_idxs)) // 1)
            indexes_to_update = random.sample(actionable_idxs, num_items_to_alter)
        elif token_strategy == "random_between_global":
            num_items_to_alter = int((selection_percent * len(tokens)) // 1)
            actionable_idxs = [x if (tokens[x] in idx_by_first.keys() or tokens[x+1] in idx_by_second.keys()) else None for x in list(range(len(tokens)-1))]
            actionable_idxs = [x for x in actionable_idxs if x is not None]
            num_items_to_alter = min(num_items_to_alter, len(actionable_idxs))
            indexes_to_update = random.sample(actionable_idxs, num_items_to_alter)
        else:
            raise NotImplementedError(f"Token selection strategy \'{token_strategy}\' is not implemented!")

        indexes_to_update.sort()

        #How we choose disfluencies for the selected tokens
        for idx in indexes_to_update:
            first_w = tokens[idx]
            if idx + 1 < len(tokens):
                second_w = tokens[idx + 1]

            if disf_strategy == "random":
                possible_dis = OrderedDict()
                if first_w in idx_by_first.keys():
                    possible_dis.update(idx_by_first[first_w])
                if idx + 1 < len(tokens) and second_w in idx_by_second.keys():
                    possible_dis.update(idx_by_second[second_w])

                norm_vals = [x/sum(list(possible_dis.values())) for x in list(possible_dis.values())]
                d_choice = list(possible_dis.keys())[np.random.choice(list(range(len(possible_dis.keys()))), p=norm_vals)]

                if d_choice[-1][0] == "(" and d_choice[-1][-1] == ")": #TODO: replace with item matching instead of string checking
                    d_choice = d_choice[-1]
                else:
                    d_choice = d_choice[0]
                d_locations.append(idx)
                d_choices.append(d_choice)

            else:
                raise NotImplementedError(f"Disfluency selection strategy \'{disf_strategy}\' is not implemented!")

        out_string = apply_disfluencies(tokens, d_locations, d_choices)
        write_file.write(out_string + "\n")




def process_birgrams():
    flat_disfluencies = load_disfluencies()
    proc_corpus = load_corpus()
    ct_total_bigrams, ct_total_nonbigrams, all_bigrams, non_bigrams = learn_bigrams(proc_corpus, flat_disfluencies)

    test_in_path = "./data/test_input.txt"
    output_path = "./data/test_output.txt"
    selection_percent = 0.5
    gen_bigrams(selection_percent, all_bigrams, test_in_path, output_path)


process_birgrams()