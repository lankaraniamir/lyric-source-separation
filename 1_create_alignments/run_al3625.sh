#!/usr/bin/env bash

stage=2
. ./cmd.sh
. ./path.sh

# stage = 0
# Prepares the musdb data to be in a format processable by Kaldi
if [ $stage -le 0 ]; then
    conda activate mirdata
    python local/prep_data_musdb_hq_al3625.py
    conda deactivate
fi

# stage = 1
# Remixes songs in the dataset while finding lyric positions to supplement data
if [ $stage -le 1 ]; then
    conda activate demucs
    python local/mashup_with_vox_morph_mono_al3625.py #conda
    conda deactivate
fi

# stage = 2
# Preps mashup data to also be in Kaldi format
if [ $stage -le 2 ]; then
    conda activate mirdata
    python local/prep_data_mashups_al3625.py
    conda deactivate
fi

# Goes through each song in each of the datasets &
# extracts its posteriorgrams and alignments using the align_and_post script
# Skips data already processed allowing processing to be picked up and stopped
# as needed
if [ $stage -le 3 ]; then
    conda activate ASA
    musdb_dir=./db/musdbhq_mono
    musdb_train_dir=${musdb_dir}/train
    musdb_dev_dir=${musdb_dir}/dev
    musdb_test_dir=${musdb_dir}/test
    mashup_train_dir=./db/mashups_mono/train
    for set_dir in $musdb_train_dir $musdb_dev_dir $musdb_test_dir $mashup_train_dir; do
        set_name=$(basename -- $set_dir)
        for song_dir in $(find ${set_dir} -maxdepth 1 -mindepth 1 -type d); do
            song_name=$(basename -- $song_dir)
            wavpath=${song_dir}/mixture.wav
            lyricspath=${song_dir}/lyrics_unsynced.txt
            timed_lyricspath=${song_dir}/lyrics_timed.txt
            if [ $set_dir == $mashup_train_dir ]; then
                savepath=results/${set_name}/${song_name}
            else
                savepath=results/${set_name}/${song_name}
            fi

            start=`date +%s`
            echo
            echo "Processing new mixture"
            echo "set_name: "${set_name}
            echo "song_name: "${song_name}
            echo "wav from: "${wavpath}
            echo "lyric from: "${lyricspath}
            echo "output path: "${savepath}
            if [ -f "${savepath}/phone_post.npy" ]; then
                echo
                echo "Already processed posteriorgram for ${song_name}.   Moving on."

            # Do not process audio without lyrics since not useful for study
            elif [ ! -s ${lyricspath} ]; then
                echo
                echo "No lyrics for ${song_name}.  Moving on."
            else

                . ./align_and_post_al3625.sh $wavpath $lyricspath $savepath ${set_name}_${song_name} $timed_lyricspath

                # Bring in the audio files here so all separation data is one place
                if [ ${set_dir} == ${mashup_train_dir} ]; then
                    cp "${mashup_train_dir}/${song_name}/mixture.wav" "${savepath}/mixture.wav"
                    cp "${mashup_train_dir}/${song_name}/vocals.wav" "${savepath}/vocals.wav"
                else
                    cp "${song_dir}/mixture.wav" "${savepath}/mixture.wav"
                    cp "${song_dir}/vocals.wav" "${savepath}/vocals.wav"
                fi
            fi

            end=`date +%s`
            runtime=$((end-start))
            echo
            echo $runtime
        done
    done
    conda deactivate
fi

echo
echo
echo
echo "***                                        ***"
echo "*** ***                                *** ***"
echo "*** *** *** FINISHED PROCESSING ALL*** *** ***"
echo "*** ***                                *** ***"
echo "***                                        ***"
echo