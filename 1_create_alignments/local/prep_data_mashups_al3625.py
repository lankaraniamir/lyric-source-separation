import os
import musdb
import string
import shutil
import csv
from collections import defaultdict
import pandas as pd
import soundfile as sf
import random

# IN_AUDIO = "./db/musdb18/"
# mus = musdb.DB(root=IN_AUDIO)
# AUDIO = "./db/mashups_all/"
IN_DIR = "db/mashups_mono/"
mus = musdb.DB(root=IN_DIR, is_wav=True)
# IN_DATA = "./db/musdb_lyrics/"
OUT_DATA = "data/"

# can add bass, drums, other, accompaniement, or any mixture of these
sources = ["vocals", "mixture"]
# in_groups = ["train", "dev"]
# out_groups = ["train", "dev"]
in_groups = ["train"]
out_groups = ["train"]

print("\n***Begin prepping mashups and its lyric files***")
print("Deleting old data")

# # Remove old folders
if not os.path.isdir(OUT_DATA):
    os.mkdir(OUT_DATA)
for group in out_groups:
    for source in sources:
        out_data_dir = OUT_DATA + group + "_mashups_" + source + "/"
        if not os.path.isdir(out_data_dir):
            os.mkdir(out_data_dir)
        else:
            for file in os.listdir(out_data_dir):
                shutil.rmtree(out_data_dir+file) if os.path.isdir(out_data_dir+file) else os.remove(out_data_dir + file)

utt_count = 0
cols_dict = defaultdict(lambda: [])
bad_punctuation = string.punctuation.replace("'", "")

for group in in_groups:
    dir_group = IN_DIR + group + "/"
    out_data_dir_group = OUT_DATA + group + "_mashups/"

    for folder in os.listdir(dir_group):
        if not os.path.isdir(dir_group+folder):
            continue

        name = folder
        song_folder = os.path.join(dir_group, name) + "/"

        # in_audio_folder = name + "/"

        parts = name.split("_")
        artist, song, sr = parts[0], parts[1], 44100

        spk_id = ("mashups_" + artist[:2].rjust(2, "0") +
                  "-part_" + song[:4].rjust(4, "0"))
        # print(spk_id)

        # modified sf.write method from musdb_lyric's musd_lyrics_cut_audio
        in_data_file = "lyrics.txt"
        mix_file = os.path.join(song_folder, "mixture.wav")
        vox_file = os.path.join(song_folder, "vocals.wav")

        print(name, " ::: SUCCESS")

        # Ordered for best visibility
        ulyr_path = song_folder + "lyrics_unsynced.txt"
        if os.path.isfile(ulyr_path):
            os.remove(ulyr_path)
            need_ulyr = True
            # need_ulyr = False
        else:
            need_ulyr = True
        if not os.path.isfile(ulyr_path):
            f_ulyrs = open(ulyr_path, "w")
            need_ulyr = True

        seglyr_path = song_folder + "lyrics_timed.txt"
        if os.path.isfile(seglyr_path):
            os.remove(seglyr_path)
            need_seglyr = True
            # need_seglyr = False
        else:
            need_seglyr = True
        if not os.path.isfile(seglyr_path):
            f_seglyrs = open(seglyr_path, "w")
            need_seglyr = True

        for utt in open(song_folder + in_data_file, "r").readlines():
            if utt_count >  99999999:
                print("Too many files! Please edit the ID tagging to compensate")
                quit()
            traits = utt.split(" ")
            cols_dict["set"].append(group)
            cols_dict["artist"].append(artist)
            cols_dict["song"].append(song)

            if traits[2] == "<UNK>" or traits[2] == "<UNK>\n":
                lyrics = "<UNK>"
            elif traits[2] == "<NO_VOX>" or traits[2] == "<NO_VOX>\n":
                lyrics = "<NO_VOX>"
            else:
                lyrics = " ".join(traits[2:]).upper().strip("\n")
                lyrics = lyrics.translate(str.maketrans("", "", bad_punctuation))
                lyrics = " ".join(lyrics.split())
                if need_ulyr:
                    # lyrics = " ".join(traits[2:])
                    f_ulyrs.write(" ".join(traits[2:]))
                if need_seglyr:
                    f_seglyrs.write(f"{start} {stop} {lyrics}\n")

            # lyrics = " ".join(traits[2:])
            cols_dict["lyrics"].append(lyrics)
            # if need_ulyr:
            #     # lyrics = " ".join(traits[2:])
            #     f_ulyrs.write(" ".join(traits[2:]))

            start = float(traits[0])
            stop = float(traits[1])
            start = "{:07.2f}".format(start)
            stop = "{:07.2f}".format(stop)
            cols_dict["start"].append(start)
            cols_dict["stop"].append(stop)

            cols_dict["spk_id"].append(spk_id)
            cols_dict["utt_id"].append(spk_id + "-" + str(utt_count).rjust(7, "0"))
            # cols_dict["mix_spk_id"].append("mix_" + spk_id)
            # cols_dict["mix_utt_id"].append("mix_" + spk_id + "-" + str(utt_count).rjust(7, "0"))
            # cols_dict["vox_spk_id"].append("vox_" + spk_id)
            # cols_dict["vox_utt_id"].append("vox_" + spk_id + "-" + str(utt_count).rjust(7, "0"))

            cols_dict["mix_file"].append(mix_file)
            cols_dict["vox_file"].append(vox_file)
            cols_dict["sr"].append(sr)
            utt_count += 1
        if need_ulyr:
            f_ulyrs.close()
        if need_seglyr:
            f_seglyrs.close()

# # print("there")
# quit()
df = pd.DataFrame(cols_dict)
all_data_file = IN_DIR + "mashups_all_data.csv"
df.to_csv(all_data_file)

for group in out_groups:
    # out_dir = OUT_DATA + group + "_musdb/"
    group_df = (df[df["set"]==group]).copy()

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

    mix_dir = OUT_DATA + group + "_mashups_mixture/"
    vox_dir = OUT_DATA + group + "_mashups_vocals/"

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

print("***Finished prepping mashups and its lyric files***\n")