import os
import musdb
import string
import shutil
import csv
from collections import defaultdict
import pandas as pd
import soundfile as sf
import random
import argparse
import re

# IN_AUDIO = "./db/musdb18/"
# mus = musdb.DB(root=IN_AUDIO)
AUDIO = "db/musdbhq/"
mus = musdb.DB(root=AUDIO, is_wav=True)
MONO_AUDIO = "db/musdbhq_mono/"
IN_DATA = "db/musdb_lyrics/"
# OUT_AUDIO = "./db/musdb_split/"
OUT_DATA = "data/"


# can add bass, drums, other, accompaniement, or any mixture of these
sources = ["vocals", "mixture"]
in_groups = ["train", "test"]
out_groups = ["train", "test", "dev"]

# Copied from musdb lyrics processing file which
# is same as intended dev set from creators of musdb
dev_tracks = ['Actions - One Minute Smile',
                'Clara Berry And Wooldog - Waltz For My Victims',
                'Johnny Lokke - Promises & Lies',
                'Patrick Talbot - A Reason To Leave',
                'Triviul - Angelsaint',
                'Alexander Ross - Goodbye Bolero',
                'Fergessen - The Wind',
                'Leaf - Summerghost',
                'Skelpolu - Human Mistakes',
                'Young Griffo - Pennies',
                'ANiMAL - Rockshow',
                'James May - On The Line',
                'Meaxic - You Listen',
                'Traffic Experiment - Sirens']


print("\n***Begin prepping musdb and its lyric files***")
print("Deleting old data")

# Borrowed from data_preparation of ASA
lex_words = []
oov_words = []
model_lexicon = "conf/dict/lexicon_raw.txt"
with open(model_lexicon,'r') as l:
    for line in l.readlines():
        lex_words.append(line.split(' ')[0])



# # Remove old folders
if not os.path.isdir(MONO_AUDIO):
    os.mkdir(MONO_AUDIO)
if not os.path.isdir(OUT_DATA):
    os.mkdir(OUT_DATA)
if os.path.isdir(IN_DATA + "dev"):
    for file in os.listdir(IN_DATA + "dev"):
        shutil.move(IN_DATA+"dev/"+file, IN_DATA+"train/"+file)
for group in out_groups:
    data_dir_group = IN_DATA + group
    audio_dir_group = MONO_AUDIO + group + "/"
    if not os.path.isdir(audio_dir_group):
        os.mkdir(audio_dir_group)
    # if not os.path.isdir(data_dir_group + "_lyrics_unsynced/"):
    #     os.mkdir(data_dir_group + "_lyrics_unsynced/")
    # if not os.path.isdir(data_dir_group + "_lyrics/"):
    #     os.mkdir(data_dir_group + "_lyrics/")
    for source in sources:
        out_data_dir = OUT_DATA + group + "_musdbhq_" + source + "/"
        if not os.path.isdir(out_data_dir):
            os.mkdir(out_data_dir)
        else:
            for file in os.listdir(out_data_dir):
                shutil.rmtree(out_data_dir+file) if os.path.isdir(out_data_dir+file) else os.remove(out_data_dir + file)


# dev_dir = AUDIO + "dev/"
# train_dir = AUDIO + "train/"
# for folder in os.listdir(dev_dir):
#     shutil.move(os.path.join(dev_dir,folder), os.path.join(train_dir, folder))

utt_count = 0
cols_dict = defaultdict(lambda: [])
audio_dir_dev = MONO_AUDIO + "dev/"
out_data_dir_dev = OUT_DATA + "dev_musdbhq/"
bad_punctuation = string.punctuation.replace("'", "")

for group in in_groups:
    in_data_dir = IN_DATA + group + "_lyrics/"
    audio_dir_group = MONO_AUDIO + group + "/"
    stereo_audio_dir_group = AUDIO + group + "/"

    out_data_dir_group = OUT_DATA + group + "_musdbhq/"
    in_data_files = os.listdir(in_data_dir)
    tracks = mus.load_mus_tracks(subsets=[group])
    for track in tracks:
        name = track.name
        in_data_file = name + ".txt"
        if in_data_file in in_data_files:
            print(name, " ::: SUCCESS")
        else:
            print(name, " ::: FALURE")
            continue

        if name in dev_tracks:
            true_group = "dev"
            audio_dir = audio_dir_dev
            out_data_dir = out_data_dir_dev
            song_folder = os.path.join(audio_dir_dev + name)
        else:
            true_group = group
            audio_dir = audio_dir_group
            out_data_dir = out_data_dir_group
            song_folder = os.path.join(audio_dir_group + name)

        song_folder_old = song_folder
        song_folder = song_folder.replace(' ', '_').replace('&', 'and').replace("'", "").replace(",", "").replace("(", "").replace(")", "")
        if not os.path.isdir(song_folder):
            os.makedirs(song_folder)


        mix_file =  os.path.join(song_folder, "mixture.wav")
        vox_file =  os.path.join(song_folder, "vocals.wav")
        lyr_path =  os.path.join(song_folder, "lyrics.txt")
        ulyr_path = os.path.join(song_folder, "lyrics_unsynced.txt")
        seglyr_path = os.path.join(song_folder, "lyrics_timed.txt")
        rawlyr_path = os.path.join(song_folder, "lyrics.raw.txt")
        need_seglyr = False
        need_ulyr = False
        need_rawlyr = False
        if os.path.isfile(ulyr_path):
            os.remove(ulyr_path)
        if not os.path.isfile(mix_file):
            sf.write(mix_file, .5*(track.audio[...,0] + track.audio[...,1]), samplerate=track.rate)
        if not os.path.isfile(vox_file):
            sf.write(vox_file, .5*(track.targets["vocals"].audio[...,0] + track.targets["vocals"].audio[...,1]), samplerate=track.rate)

        new_stereo = os.path.join(song_folder, "stereo")
        if not os.path.isdir(new_stereo):
            os.mkdir(new_stereo)
            stereo_mix_file =  os.path.join(new_stereo, "mixture.wav")
            stereo_vox_file =  os.path.join(new_stereo, "vocals.wav")
            stereo_mix_old = os.path.join(stereo_audio_dir_group + name, "mixture.wav")
            stereo_vox_old = os.path.join(stereo_audio_dir_group + name, "vocals.wav")
            shutil.copyfile(stereo_mix_old, stereo_mix_file)
            shutil.copyfile(stereo_vox_old, stereo_vox_file)

        if not os.path.isfile(lyr_path):
            shutil.copy(in_data_dir+in_data_file, lyr_path)
        if not os.path.isfile(seglyr_path):
            f_seglyrs = open(seglyr_path, "w")
            need_seglyr = True
        if not os.path.isfile(ulyr_path):
            f_ulyrs = open(ulyr_path, "w")
            need_ulyr = True
        if not os.path.isfile(rawlyr_path):
            f_rawlyr = open(rawlyr_path, "w")
            need_rawlyr = True

        artist, song, sr = track.artist, track.title, track.rate
        spk_id = (artist[:10].replace(" ", "_").ljust(10, "_") +
                  "-" + song[:10].replace(" ", "_").ljust(10, "_"))

        for utt in open(in_data_dir + in_data_file, "r").readlines():
            if utt_count >  99999999:
                print("Too many files! Please edit the ID tagging to compensate")
                quit()
            traits = utt.split(" ")
            cols_dict["set"].append(true_group)
            cols_dict["artist"].append(artist)
            cols_dict["song"].append(song)

            lyrics = None
            if traits[0] == "*":
                lyrics="<UNK>"
                traits.pop(0)

            cols_dict["vox_type"].append(traits[2])
            start = int(traits[0][3:5]) + 60*int(traits[0][0:2])
            start = str(start).rjust(4, "0")
            stop = int(traits[1][3:5]) + 60*int(traits[1][0:2])
            stop = str(stop).rjust(4, "0")
            cols_dict["start"].append(start)
            cols_dict["stop"].append(stop)

            if lyrics != "<UNK>" and traits[2] == "d":
                lyrics="<NO_VOX>"
            elif lyrics != "<UNK>":
                lyrics = " ".join(traits[3:]).upper().strip("\n")
                lyrics = lyrics.translate(str.maketrans("", "", bad_punctuation))
                lyrics = lyrics.split()
                # Also borrowed from data prep of ASA
                for word in lyrics:
                    if not word in lex_words:
                        oov_words.append(word)
                        lex_words.append(word)
                lyrics = " ".join(lyrics)
                # if need_seglyr:
                #     f_seglyrs.write(f"{start} {stop} {lyrics}\n")
                if need_ulyr:
                    f_ulyrs.write(lyrics + "\n")
                if need_rawlyr:
                    f_rawlyr.write(lyrics + " ")
            cols_dict["lyrics"].append(lyrics)
            # if need_ulyr:
            #     f_ulyrs.write(lyrics + "\n")

            cols_dict["spk_id"].append(spk_id)
            cols_dict["utt_id"].append(spk_id + "-" + str(utt_count).rjust(7, "0"))
            cols_dict["mix_file"].append(mix_file)
            cols_dict["vox_file"].append(vox_file)
            cols_dict["sr"].append(sr)


            utt_count += 1

        if need_seglyr:
            f_seglyrs.close()
        if need_ulyr:
            f_ulyrs.close()
        if need_rawlyr:
            f_rawlyr.close()
        # if true_group == "dev":
        #     shutil.move(in_data_dir+in_data_file, IN_DATA+"dev_lyrics/"+in_data_file)
        if os.path.isdir(song_folder_old):
            shutil.rmtree(song_folder_old)

        #

# # print("there")
# quit()
df = pd.DataFrame(cols_dict)
all_data_file = AUDIO + "musdbhq_all_data.csv"
df.to_csv(all_data_file)

for group in out_groups:
    group_df = (df[df["set"]==group]).copy()
    # for source in sources:

    text = group_df.reindex(columns=["utt_id", "lyrics"])
    segments = group_df.reindex(columns=["utt_id", "spk_id", "start", "stop"])
    utt2spk = group_df.reindex(columns=["utt_id", "spk_id"])
    spk2utt = group_df.reindex(columns=["spk_id", "utt_id"])
    mix_wav_scp = group_df.reindex(columns=["spk_id", "mix_file"])
    vox_wav_scp = group_df.reindex(columns=["spk_id", "vox_file"])

    text.sort_values(["utt_id", "lyrics"], inplace=True)
    segments.sort_values(["utt_id", "spk_id", "start", "stop"], inplace=True)
    utt2spk.sort_values(["utt_id", "spk_id"], inplace=True)
    spk2utt.sort_values(["spk_id", "utt_id"], inplace=True)
    mix_wav_scp.sort_values(["spk_id", "mix_file"], inplace=True)
    vox_wav_scp.sort_values(["spk_id", "vox_file"], inplace=True)

    mix_dir = OUT_DATA + group + "_musdbhq_mixture/"
    vox_dir = OUT_DATA + group + "_musdbhq_vocals/"

    # wav_scp.to_csv(out_dir + "wav.scp", index=None, header=None, sep=' ', mode='w', quoting=csv.QUOTE_NONE,escapechar=' ')
    # segments.to_csv(out_dir + "segments", index=None, header=None, sep=' ', mode='w', quoting=csv.QUOTE_NONE,escapechar=' ')
    # utt2spk.to_csv(out_dir + "utt2spk", index=None, header=None, sep=' ', mode='w', quoting=csv.QUOTE_NONE,escapechar=' ')

    f_mix = open(mix_dir + "text", "w")
    f_vox = open(vox_dir + "text", "w")
    for utt, lyrics in zip(text.utt_id, text.lyrics):
        f_mix.write(utt + " " + lyrics + "\n")
        f_vox.write(utt + " " + lyrics + "\n")
    f_mix.close()
    f_vox.close()

    f_mix = open(mix_dir + "segments", "w")
    f_vox = open(vox_dir + "segments", "w")
    for utt, spk_id, start, stop in zip(segments.utt_id, segments.spk_id, segments.start, segments.stop):
        f_mix.write(utt + " " + spk_id + " " + start + " " + stop + "\n")
        f_vox.write(utt + " " + spk_id + " " + start + " " + stop + "\n")
    f_mix.close()
    f_vox.close()

    f_mix = open(mix_dir + "utt2spk", "w")
    f_vox = open(vox_dir + "utt2spk", "w")
    for utt, spk_id in zip(utt2spk.utt_id, utt2spk.spk_id):
        f_mix.write(utt + " " + spk_id + "\n")
        f_vox.write(utt + " " + spk_id + "\n")
    f_mix.close()
    f_vox.close()

    f_mix = open(mix_dir + "spk2utt", "w")
    f_vox = open(vox_dir + "spk2utt", "w")
    speakers = spk2utt.spk_id.unique()
    for speaker in speakers:
        buffer = (spk2utt.loc[spk2utt["spk_id"] == speaker]).utt_id
        f_mix.write(speaker + " " + " ".join(buffer) + "\n")
        f_vox.write(speaker + " " + " ".join(buffer) + "\n")
    f_mix.close()
    f_vox.close()

    f_mix = open(mix_dir + "wav.scp", "w")
    f_vox = open(vox_dir + "wav.scp", "w")
    speakers = mix_wav_scp.spk_id.unique()
    for speaker in speakers:
        mix = (mix_wav_scp.loc[mix_wav_scp["spk_id"] == speaker]).mix_file.unique()[0]
        vox = (vox_wav_scp.loc[vox_wav_scp["spk_id"] == speaker]).vox_file.unique()[0]
        f_mix.write(speaker + " " + mix + "\n")
        f_vox.write(speaker + " " + vox + "\n")
    f_mix.close()
    f_vox.close()

    # f = open(out_dir + "segments", "w")

    # f = open(out_dir + "text", "w")

    # f = open(out_dir + "spk2utt", "w")
oov_dir = "data/oov/"
if not os.path.isdir(oov_dir):
    os.mkdir(oov_dir)
oov_path = os.path.join(oov_dir, "oov_words.txt")

with open(oov_path,'w') as oov:
    for word in oov_words:
        oov.write(word + '\n')
print("***Finished prepping musdb and its lyric files***\n")