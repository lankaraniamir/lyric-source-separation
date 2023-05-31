import os
import csv
import shutil
import pathlib
import pandas as pd
from collections import defaultdict

# SPLIT_AUDIO = "./db/musdb_split/"
# # print("***creating necessary dictionary files from musdb lyrics***\n")
# df = pd.read_csv(SPLIT_AUDIO + "musdb_all_data.csv")
# lyrics = df.edited_lyrics

RAW_LEXICON = "./db/musdb_lyrics/words_and_phonemes.txt"
OUT_LEXICON_PARENT = "./data/local/"
OUT_LEXICON_DIR = "./data/local/dict/"

# if os.path.isdir(OUT_AUDIO):
#     shutil.rmtree(OUT_LEXICON_DIR)

path = pathlib.Path(OUT_LEXICON_DIR)
path.parent.mkdir(exist_ok=True, parents=True)
for file in os.listdir(OUT_LEXICON_PARENT):
    if not os.path.isdir(OUT_LEXICON_PARENT + file):
        os.remove(OUT_LEXICON_PARENT + file)
for file in os.listdir(OUT_LEXICON_DIR):
    os.remove(OUT_LEXICON_DIR + file)

f = open(OUT_LEXICON_DIR + "silence_phones.txt", "w")
silence_phones = ["SPN", "NSN", "SIL"]
for phone in silence_phones:
    f.write(phone + "\n")
f.close()

nonsilence_dict = defaultdict(lambda: [])
for lexical_pair in open(RAW_LEXICON, "r").readlines():
    split_pair = lexical_pair.split("\t")
    word = split_pair[0].split("(")[0]
    phoneme = split_pair[1].split("\n")[0]
    nonsilence_dict["word"].append(word)
    nonsilence_dict["phoneme"].append(phoneme)
nonsilence_dict["word"] = [word.split("(")[0] for word in nonsilence_dict["word"]]
# nonsilence_phones = pd.DataFrame(nonsilence_dict)
# nonsilence_phones.sort_values(["word", "phoneme"], inplace=True)
# f = open(OUT_LEXICON_DIR + "nonsilence_phones.txt", "w")
# for word, phoneme in zip(nonsilence_phones.word, nonsilence_phones.phoneme):
#     f.write(word + " " + phoneme + "\n")
# f.close()

f = open(OUT_LEXICON_DIR + "nonsilence_phones.txt", "w")
all_nonsil_phones = " ".join(nonsilence_dict["phoneme"])
unique_nonsil_phones = sorted(set(all_nonsil_phones.split(" ")))
for phone in unique_nonsil_phones:
    f.write(phone + "\n")
f.close()

lexicon_dict = nonsilence_dict
lexicon_dict["word"].extend(["<UNK>", "<NO_VOX>", "<SIL>"])
lexicon_dict["phoneme"].extend(["SPN", "NSN", "SIL"])
# lexicon_dict["word"].extend(["<UNK>", "<MIXED>", "<NO_VOX>", "<SIL>"])
# lexicon_dict["phoneme"].extend(["SPN", "SPN", "NSN", "SIL"])
lexicon = pd.DataFrame(lexicon_dict)
lexicon.sort_values(["word", "phoneme"], inplace=True)
f = open(OUT_LEXICON_DIR + "lexicon.txt", "w")
for word, phoneme in zip(lexicon.word, lexicon.phoneme):
    f.write(word + " " + phoneme + "\n")
f.close()

f = open(OUT_LEXICON_DIR + "optional_silence.txt", "w")
f.write("SIL\n")
# output = "SIL".encode("ascii")
# f.write("output")
f.close()