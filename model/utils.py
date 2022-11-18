import json
import pickle

START_TOKEN = "[START]"
END_TOKEN = "[END]"
PAD_TOKEN = "[PAD]"


def load_disfluencies(d_path="./data/disfluency_key.json", acceptable_dis = ["Pause", "Vocal Noises", "Extra"]):
    disfls = open(d_path, "r")
    data = json.load(disfls)

    flat_disfluencies = []
    for d_k in acceptable_dis:
        #temp = ["(" + d_k.replace(" ", "_").lower() + "_" + x.replace("(", "") for x in list(data[d_k].values())]
        flat_disfluencies = flat_disfluencies + list(data[d_k].values())
    return flat_disfluencies


def load_corpus(c_path="./data/output.txt"):
    file_corpus = open(c_path, "r")
    proc_corpus = file_corpus.readlines()
    return proc_corpus


def save_model(model, path='model.pt'):
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path='model.pt', device='cpu'):
    with open(path, 'rb') as file:
        model = pickle.load(file)
        return model.to(device)