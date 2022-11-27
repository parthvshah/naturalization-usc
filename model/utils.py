import json
import pickle
import re

START_TOKEN = "[START]"
END_TOKEN = "[END]"
PAD_TOKEN = "[PAD]"


def load_disfluencies(
    d_path="./data/disfluency_key.json",
    acceptable_dis=["Pause", "Vocal Noises", "Extra"],
):
    disfls = open(d_path, "r")
    data = json.load(disfls)

    flat_disfluencies = []
    for d_k in acceptable_dis:
        # temp = ["(" + d_k.replace(" ", "_").lower() + "_" + x.replace("(", "") for x in list(data[d_k].values())]
        flat_disfluencies = flat_disfluencies + list(data[d_k].values())
    return flat_disfluencies


def load_corpus(c_path="./data/output.txt", clean_filters=["newlines"]):
    file_corpus = open(c_path, "r")
    proc_corpus = file_corpus.readlines()
    clean_corpus = []
    if clean_filters is not None and len(clean_filters) > 0:
        for line in proc_corpus:
            if "newlines" in clean_filters:
                clean_corpus.append(re.subn("/\r?\n|\r/", "", line.strip())[0])

    return clean_corpus


def save_model(model, seq_len, path='model.pt'):
    with open(path, 'wb') as file:
        model_dict = {"seq_len": seq_len, "model": model}
        pickle.dump(model_dict, file)


def load_model(path='model.pt', device='cpu'):
    with open(path, 'rb') as file:
        model_dict = pickle.load(file)
        return model_dict["model"].to(device), model_dict["seq_len"]

def filter_raw(raw_path, filtered_out_path):
    raw_file = open(raw_path, "r")
    write_file = open(filtered_out_path, "w+")


    for line in raw_file.readlines():
        tokens = line.split()
        new_s = ""
        for token in tokens:
            if not (token[0] == "(" and token[-1] == ")"):
                new_s = new_s + " " + token
        write_file.write(new_s + "\n")
