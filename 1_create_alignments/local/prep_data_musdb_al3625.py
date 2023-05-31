import os
import musdb
import string
import shutil
import csv
from collections import defaultdict
import pandas as pd
import soundfile as sf
import random

IN_AUDIO = "./db/musdb18/"
mus = musdb.DB(root=IN_AUDIO)
# IN_AUDIO = "./db/musdb18hq/"
# mus = musdb.DB(root=IN_AUDIO, is_wav=True)
IN_DATA = "./db/musdb_lyrics/"
OUT_AUDIO = "./db/musdb_split/"
OUT_DATA = "./data/"

# can add bass, drums, other, accompaniement, or any mixture of these
sources = ["vox", "mix"]
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

# Switch to this later
# temp_tracks = mus.load_mus_tracks(subsets=["test"])
# dev_tracks = [temp_tracks[index] for index in random.sample(range(0,
# len(temp_tracks)-1), int(2/3*len(temp_tracks)/2))]

print("\n***Begin prepping musdb and its lyric files***")
print("Deleting old data")

# Remove old folders
if os.path.isdir(OUT_AUDIO):
    shutil.rmtree(OUT_AUDIO)
if not os.path.isdir(OUT_DATA):
    os.mkdir(OUT_DATA)
os.mkdir(OUT_AUDIO)
for group in out_groups:
    out_audio_dir = OUT_AUDIO + group + "/"
    os.mkdir(out_audio_dir)
    for source in sources:
        os.mkdir(out_audio_dir + source + "/")
        out_data_dir = OUT_DATA + group + "_musdb_" + source + "/"
        if not os.path.isdir(out_data_dir):
            os.mkdir(out_data_dir)
        else:
            for file in os.listdir(out_data_dir):
                os.remove(out_data_dir + file)

utt_count = 0
cols_dict = defaultdict(lambda: [])
out_data_dir_dev = OUT_DATA + "dev_musdb/"
out_audio_dir_dev = OUT_AUDIO + "dev/"
bad_punctuation = string.punctuation.replace("'", "")

for group in in_groups:
    in_data_dir = IN_DATA + group + "_lyrics/"
    in_audio_dir = IN_AUDIO + group + "/"
    out_data_dir_group = OUT_DATA + group + "_musdb/"
    out_audio_dir_group = OUT_AUDIO + group + "/"

    in_data_files = os.listdir(in_data_dir)
    tracks = mus.load_mus_tracks(subsets=[group])
    for track in tracks:
        in_file_no_ext = track.name
        in_data_file = in_file_no_ext + ".txt"
        in_audio_file = in_file_no_ext + ".mp4"
        out_audio_file = in_file_no_ext.replace(" ", "_") + ".wav"

        if in_data_file in in_data_files:
            print(in_file_no_ext, " ::: SUCCESS")
        else:
            print(in_file_no_ext, " ::: FALURE")
            continue

        if in_file_no_ext in dev_tracks:
            true_group = "dev"
            out_data_dir = out_data_dir_dev
            out_audio_dir = out_audio_dir_dev
        else:
            true_group = group
            out_data_dir = out_data_dir_group
            out_audio_dir = out_audio_dir_group

        artist, song, sr = track.artist, track.title, track.rate

        spk_id = (artist[:10].replace(" ", "_").ljust(10, "_") +
                  "-" + song[:10].replace(" ", "_").ljust(10, "_"))

        # modified sf.write method from musdb_lyric's musd_lyrics_cut_audio
        mix_file = os.path.join(out_audio_dir, "mix", out_audio_file)
        vox_file = os.path.join(out_audio_dir, "vox", out_audio_file)
        sf.write(mix_file, track.audio, samplerate=track.rate)
        sf.write(vox_file, track.targets["vocals"].audio, samplerate=track.rate)

        # Ordered for best visibility
        for utt in open(in_data_dir + in_data_file, "r").readlines():
            if utt_count >  99999999:
                print("Too many files! Please edit the ID tagging to compensate")
                quit()
            traits = utt.split(" ")
            cols_dict["set"].append(true_group)
            cols_dict["artist"].append(artist)
            cols_dict["song"].append(song)

            if traits[0] == "*":
                cols_dict["lyrics"].append("<UNK>")
                traits.pop(0)
            elif traits[2] == "d":
                cols_dict["lyrics"].append("<NO_VOX>")
            else:
                lyrics = " ".join(traits[3:]).upper().strip("\n")
                lyrics = lyrics.translate(str.maketrans("", "", bad_punctuation))
                lyrics = " ".join(lyrics.split())
                cols_dict["lyrics"].append(lyrics)
            cols_dict["vox_type"].append(traits[2])

            start = int(traits[0][3:5]) + 60*int(traits[0][0:2])
            start = str(start).rjust(4, "0")
            stop = int(traits[1][3:5]) + 60*int(traits[1][0:2])
            stop = str(stop).rjust(4, "0")
            cols_dict["start"].append(start)
            cols_dict["stop"].append(stop)

            cols_dict["spk_id"].append(spk_id)
            cols_dict["utt_id"].append(spk_id + "-" + str(utt_count).rjust(7, "0"))
            cols_dict["mix_file"].append(mix_file)
            cols_dict["vox_file"].append(vox_file)
            cols_dict["sr"].append(sr)
            utt_count += 1

df = pd.DataFrame(cols_dict)
all_data_file = OUT_AUDIO + "musdb_all_data.csv"
df.to_csv(all_data_file)

for group in out_groups:
    # out_dir = OUT_DATA + group + "_musdb/"
    group_df = (df[df["set"]==group]).copy()
    for source in sources:

        wav_scp = group_df.reindex(columns=["utt_id", source + "_file"])
        text = group_df.reindex(columns=["utt_id", "lyrics"])
        segments = group_df.reindex(columns=["utt_id", "spk_id", "start", "stop"])
        utt2spk = group_df.reindex(columns=["utt_id", "spk_id"])
        spk2utt = group_df.reindex(columns=["spk_id", "utt_id"])

        wav_scp.sort_values(["utt_id", source + "_file"], inplace=True)
        segments.sort_values(["utt_id", "spk_id", "start", "stop"], inplace=True)
        utt2spk.sort_values(["utt_id", "spk_id"], inplace=True)
        text.sort_values(["utt_id", "lyrics"], inplace=True)
        spk2utt.sort_values(["spk_id", "utt_id"], inplace=True)

        out_dir = OUT_DATA + group + "_musdb_" + source + "/"
        wav_scp.to_csv(out_dir + "wav.scp", index=None, header=None, sep=' ', mode='w', quoting=csv.QUOTE_NONE,escapechar=' ')
        segments.to_csv(out_dir + "segments", index=None, header=None, sep=' ', mode='w', quoting=csv.QUOTE_NONE,escapechar=' ')
        utt2spk.to_csv(out_dir + "utt2spk", index=None, header=None, sep=' ', mode='w', quoting=csv.QUOTE_NONE,escapechar=' ')

        # text.to_csv(out_dir + "text", index=None, header=None, sep=' ', mode='w', quoting=csv.QUOTE_NONE,escapechar=' ')
        f = open(out_dir + "text", "w")
        for utt, lyrics in zip(text.utt_id, text.lyrics):
            f.write(utt + " " + lyrics + "\n")
        # spk2utt.to_csv(out_dir + "spk2utt", index=None, header=None, sep=' ', mode='w', quoting=csv.QUOTE_NONE,escapechar=' ')

        # for utt, lyric in zip(text.spk_id, text.utt_id):
        # lines = [""]*len(speakers)
        # lines = []
        speakers = spk2utt.spk_id.unique()
        f = open(out_dir + "spk2utt", "w")
        for speaker in speakers:
            buffer = (df.loc[df["spk_id"] == speaker]).utt_id
            f.write(speaker + " " + " ".join(buffer) + "\n")
            # lines.append(speaker + " " + " ".join(buffer))
            # print(lines)
            # lines[i] = speaker + " " + " ".join(buffer)
            # lines[i] = speaker + " " + " ".join(buffer)
        # f.writelines(lines)

        # spk2utt.to_csv(out_dir + "spk2utt", index=None, header=None, sep=' ', mode='w', quoting=csv.QUOTE_NONE,escapechar=' ')

print("***Finished prepping musdb and its lyric files***\n")