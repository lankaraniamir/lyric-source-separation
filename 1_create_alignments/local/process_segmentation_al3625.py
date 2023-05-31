# # from pydub import AudioSegment
# # audio = AudioSegment.from_file(file_path)
# print(audio.duration_seconds)import os, argparse, re
import ffmpeg
import pydub
import subprocess
import os
import sys
import shutil

#borrows file-reading method from ASA repository's data prep file
def main(wav_path, timed_lyrics, padding, fade):
# savepath=$3
    temp = os.path.join(os.path.dirname(wav_path), "temp.wav")
    seg_times = []
    padding = float(padding)
    fade = float(fade)
    # print("")
    # print("Times:")
    with open(timed_lyrics,'r',encoding="utf-8") as r:
        for line in r.readlines():
            words = line.split()
            start = int(words[0])
            end = int(words[1])
            if not seg_times or start - seg_times[-1][1] >= 2*padding:
                seg_times.append([start, end])
            # elif start - seg_times[-1][1] >= 0:
            else:
                seg_times[-1][1] = end


    silences = []
    wav = pydub.AudioSegment.from_file(wav_path)
    duration = wav.duration_seconds
    if seg_times[0][0] > padding:
        silences.append([0, seg_times[0][0]-padding])

    for l_interval, r_interval in zip(seg_times[:-1], seg_times[1:]):
        if r_interval[0] - l_interval[1] > 2*padding:
            silences.append([l_interval[1] + padding, r_interval[0] - padding])
    if r_interval[1] < duration - padding:
        silences.append([r_interval[1], duration])

    for silence in silences:
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-af",
        f"volume=enable='between(t,{silence[0]},{silence[1]})':volume=0", temp])
        os.replace(temp, wav_path)
        # subprocess.run(["sox", "-m",
        #                 "-t", "wav", f"|sox -V1 {wav_path} -t wav - fade t 0 {silence[0]+fade/2} {fade}",
        #                 "-t", "wav", f"|sox -V1 {wav_path} -t wav - trim {silence[0]-fade/2} fade t {fade} {silence[1] - silence[0] + fade} {fade} gain -40 pad {silence[0]-fade/2}",
        #                 "-t", "wav", f"|sox -V1 {wav_path} -t wav - trim {silence[1]-fade/2} fade t {fade} 0 0 pad {silence[1]-fade/2}",
        #                 f"|{wav_path} gain 9.542"])

if __name__ == '__main__':
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
        timed_lyrics = sys.argv[2]
        padding = sys.argv[3]
        fade = sys.argv[4]
    else:
        #Test sample
        wav_path="/home/lankaraniamir/new/kaldi-trunk/egs/new/ASA_ICASSP2021/a2l/example/preisolated_vox/EXAMPLE_vocals.wav"
        timed_lyrics="/home/lankaraniamir/new/kaldi-trunk/egs/new/ASA_ICASSP2021/a2l/example/preisolated_vox/lyrics.txt"
        padding = .7
        fade = .2
    main(wav_path, timed_lyrics, padding, fade)