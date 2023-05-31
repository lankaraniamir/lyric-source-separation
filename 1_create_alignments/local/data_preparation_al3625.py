#!/usr/bin/python

import os, argparse, re
import sys, codecs

def main(lyrics_path,wav_path,lexicon_path,save_dir):

    lex_words = []
    with open(lexicon_path,'r') as l:
        for line in l.readlines():
          lex_words.append(line.split(' ')[0])

    text = []; wavscp = []; utt2spk = []; text_norepeat = []
    spk2utt = []; segments = []
    audio_format = wav_path.split('.')[-1]
    spk_id = wav_path.split('.'+audio_format)[0].split('/')[-1]
    t = []
    with open(lyrics_path,'r',encoding="utf-8") as r:
        utt_count = 0
        utt_ids = []
        for line in r.readlines():
            if line.replace('\n','') != '':
                utt_id = spk_id + "-" + str(utt_count).rjust(3, "0")
                utt_ids.append(utt_id)
                words = line.split()
                start = words[0]
                stop = words[1]
                utt_words = " ".join(words[2:])
                lyrics_line = re.sub("[^A-Za-z0-9' ]+",'',utt_words.replace('\n',''))
                text.append(utt_id + ' ' + lyrics_line.upper())
                segments.append(f"{utt_id} {spk_id} {start} {stop}")
                utt2spk.append(utt_id + " " + spk_id)
                t.append(lyrics_line)
                utt_count += 1

        spk2utt.append(spk_id + " " + " ".join(utt_ids))
        wavscp.append(spk_id + ' sox --norm=-3 ' + wav_path +' -G -t wav -r 16000 -c 1 - remix 1 |')
        t_raw = ' '.join(t).replace('  ',' ')
        # text.append(utt_id + ' ' + t_raw.upper())
        # utt2spk.append(utt_id + ' ' + utt_id)
        # wavscp.append(utt_id + ' sox --norm=-3 ' + wav_path +' -G -t wav -r 16000 -c 1 - remix 1 |')
    # Extract Out of Vocabulary words in the test file.
    # This is necessary for extending the lexicon for alignment stage.
    oov_words = []
    for word in t_raw.split(' '):
        if not word.upper() in lex_words:
            oov_words.append(word.upper())
    with open(os.path.join(save_dir,'oov_words.txt'),'w') as oov:
        for word in oov_words:
            oov.write(word + '\n')

    with open(os.path.join(save_dir,'text'),'w') as wt, open(os.path.join(save_dir,'wav.scp'),'w') as ww, open(os.path.join(save_dir,'utt2spk'),'w') as wu, open(os.path.join(save_dir,'segments'),'w') as ws, open(os.path.join(save_dir,'spk2utt'),'w') as wuu:
        for i in range(len(text)):
            wt.write(text[i] + '\n')
            wu.write(utt2spk[i]+'\n')
            ws.write(segments[i]+'\n')
        for i in range(len(wavscp)):
            ww.write(wavscp[i]+'\n')
            wuu.write(spk2utt[i]+"\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("lyrics_path", type=str, help="Path to lyrics")
    parser.add_argument("wav_path", type=str, help="path to audio")
    parser.add_argument("lexicon_path", type=str, help="path to lexicon, i.e conf/dict/lexicon_raw.txt")
    parser.add_argument("save_dir", type=str, help="path to save the data files")

    args = parser.parse_args()

    lyrics_path = args.lyrics_path
    wav_path = args.wav_path
    lexicon_path = args.lexicon_path
    save_dir = args.save_dir
    main(lyrics_path,wav_path,lexicon_path,save_dir)
