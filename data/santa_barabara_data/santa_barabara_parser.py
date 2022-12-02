import re
import os
import json

"""
Author: Ryan Luu
Date: 9/21/2022

This program parses out the Santa Barabara Corpus and inserts disfluency tags based on the provided annotations.txt provided by the Santa_Barabara 
corpus Foundation itself. 

From that annotation list, we create a dictonary called DISFLUENCY_DICT and map out / replace occurrences of found disfluencies within the Santa Barabra Corpus
    - Extra disfluences not present in annotations.txt can be added in the global variable EXTRA_DISFLUECIES
    - If we wish for a select subset of certain disfluences, they can be specified in SELECTED_DISFLUENCIES
        - Non-selected disfluences are removed and cleaned accordingly from thet text.
    - See disfluency_key.json for the naming convention used to modify these parameters.
    
The program will then scrub over each line and replace like so:

' .. You fool ... I've been trained in your Jedi arts by Count Dooku. ' 

(Would Transform into) ======>
        
' (short_pause) You fool (medium_pause) I've been trained in your Jedi arts by Count Dooku. ' 

Two files are then outputted from this program: 
    - (sb_select_insertions_transcription.txt) - contains only identified disfluency lines
    - (sb_full_insertions_transcription.txt) - contains both identified disfluency lines and lines that don't contain disfluencies

"""

DISFLUENCY_DICT = {}
EXTRA_DISFLUENCIES = [
    ("Extra", "um", "(um)"),
    ("Extra", "umm", "(umm)"),
    ("Extra", "uh", "(uh)"),
    ("Extra", "uhh", "(uhh)"),
    ("Extra", "--", "(pause_medium)"),
    ("Extra", "Swallow", "(SWALLOW)"),
    ("Extra", "ClearThroat", "(THROAT)"),
    ("Extra", "Cough", "(COUGH)"),
]
SELECTED_DISFLUENCIES = ["Pause", "Vocal Noises", "Extra"]
translation_table = {}
DIRECTORY = "TRN"


def preprocess_disfluency_dict():
    """
    # Grabs the documentation for the SantaBarabaraCorupus and converts it into a key for our list of disfluencies
    # Data is return as a nested dict

    Example:
    ============================================================
    Units
        Intonation Unit			RETURN
        Truncated intonation unit		--
        word				SPACE
        truncated word			_

    Speakers
        Speaker identity/turn start		:
        Speech overlap			[ ]
    ============================================================
        {
        'Units': {'RETURN': '(intonation_unit)', '--': '(truncated_intonation_unit)', 'SPACE': '(word)', '_': '(truncated_word)'},
        'Speakers': {':': '(speaker_identity/turn_start)', '[ ]': '(speech_overlap)'}
        }
    ============================================================
    """

    annotations = open("annotations.txt").readlines()
    curr_section = ""
    for i in range(2, len(annotations)):  # Can skip first two lines
        line = annotations[i].strip().split("\t")
        if len(line) == 1 and line[0] != "":  # Checking if line is a section
            curr_section = line[0]
            DISFLUENCY_DICT[line[0]] = {}
        else:
            if line[0] == "":
                continue
            key = line[0].replace(" ", "_").replace("-", "_").replace("/", "_")
            section = curr_section.replace(" ", "_")
            DISFLUENCY_DICT[curr_section][line[-1]] = (
                "(" + section.lower() + "_" + key.lower() + ")"
            )
    return


def add_extra_disfluency(extra):
    """
    Function for adding extra_disfluencies that we want hand coded based on global param EXTRA_DISFLUENCIES.
    Adds to dict if outer_key exists in dictonary already. Else it creates a new entry and adds accordingly.

    Format is (outer_key, inner_key, value)
    example: (Pause, um, (um))

    """
    for disfluency in extra:
        type = disfluency[0]
        key = disfluency[1]
        value = disfluency[2]
        if type in DISFLUENCY_DICT:
            DISFLUENCY_DICT[type][key] = value
        else:
            DISFLUENCY_DICT[type] = {}
            DISFLUENCY_DICT[type][key] = value
    return


def output_disfluency_dict_to_json():
    """
    Prints out our set of disfluency keys selected as disflueny_key.json into current directory
    """
    with open("disfluency_key.json", "w") as outfile:
        json.dump(DISFLUENCY_DICT, outfile, indent=4)
    return


def output_SB_truncated_disfluency_file(output_string):
    """
    Outputs only tagged lines with disfluencies identified within.
    """
    with open("sb_select_insertions_transcription.txt", "w") as outfile:
        outfile.write(output_string)


def output_SB_full_disfluency_file(output_string):
    """
    Outputs both tagged lines with found disfluencies and lines without them.
    """
    with open("sb_full_insertions_transcription.txt", "w") as outfile:
        outfile.write(output_string)


def select_disfluencies(selection_list):
    """

    select_disfluencies makes a copy of our DISFLUENCY_DICT with the selected keys-vals passed in from selection_list.
    Every other value (either in ban_list or not in our selection list) in DISFLUENCY_DICT is then changed to '' so that we can
    replace and clean them up in the overall corpus.

    The ban_list exists to sort out outliers that messes with the overall regex or replacing tags

    :param selection_list: a list of disfluencies that we wish to grab
    :return: chosen_disfluences: a dictionary with all the same keys in DISFLUENCY_DICT but only values corresponding to selection_list are kept
    """
    chosen_disfluencies = {}
    ban_list = ["Transitional Continuity", "Speakers", "Reserved Symbols"]
    for key, val in DISFLUENCY_DICT.items():
        if key in selection_list:
            for inner_k, inner_v in val.items():
                chosen_disfluencies[inner_k] = inner_v
        elif key not in ban_list:
            for inner_k, inner_v in val.items():
                chosen_disfluencies[inner_k] = ""

    return chosen_disfluencies


def count_disfluencies():

    count_dict = {}
    for key, val in DISFLUENCY_DICT.items():
        for inner_k, inner_v in val.items():
            count_dict[inner_v] = 0

    for filename in os.listdir(DIRECTORY):
        if not filename.startswith("._"):  # Modified for working on Windows
            f = os.path.join(DIRECTORY, filename)
            if os.path.isfile(f):
                transcript = open(f).readlines()
                for line in transcript:
                    if "$" not in line:  # Avoid lines that are comments
                        try:
                            field_information = line.split("\t")
                            field_value = field_information[-1]
                        except:
                            print("Exception line occured on: " + line)
                            continue

                    for key, val in DISFLUENCY_DICT.items():
                        for inner_k, inner_v in val.items():
                            if inner_k in field_value:
                                count_dict[inner_v] += 1

    return count_dict


if __name__ == "__main__":
    preprocess_disfluency_dict()  # Grabs list of disfluencies from parsing annotations.txt
    add_extra_disfluency(EXTRA_DISFLUENCIES)  # Adds Extra handcoded disfluencies
    chosen_disfluencies = select_disfluencies(SELECTED_DISFLUENCIES)

    output_string_select = ""
    output_string_full = ""
    for filename in os.listdir(DIRECTORY):
        if not filename.startswith("._"):  # Modified for working on Windows
            f = os.path.join(DIRECTORY, filename)
            if os.path.isfile(f):
                transcript = open(f).readlines()
                for line in transcript:
                    if "$" not in line:  # Avoid lines that are comments
                        try:
                            field_information = line.split("\t")
                            field_value = field_information[-1]
                        except:
                            print("Exception line occured on: " + line)
                            continue

                        pattern = re.compile(
                            r"(?<!\w)("
                            + "|".join(
                                re.escape(key) for key in chosen_disfluencies.keys()
                            )
                            + r")(?!\w)"
                        )
                        result = pattern.sub(
                            lambda x: chosen_disfluencies[x.group()], field_value
                        )
                        if result != field_value:
                            result = result.replace("=", "").replace("YWN", "")
                            result = re.sub("[^A-Za-z()_']+", " ", result)
                            if "(" in result:
                                output_string_select += result.strip() + "\n"
                                output_string_full += result.strip() + "\n"
                            else:
                                output_string_full += result.strip() + "\n"

    output_SB_truncated_disfluency_file(output_string_select)
    output_SB_full_disfluency_file(output_string_full)
    # output_disfluency_dict_to_json()       # Uncomment to see disfluency_key.json
