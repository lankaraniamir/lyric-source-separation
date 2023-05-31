#!/bin/bash

conda activate ASA
nj=8
stage=0

. ./path.sh
. ./cmd.sh

. ./utils/parse_options.sh

wavpath=$1
lyricspath=$2
savepath=$3
rec_id=$4
timed_lyricspath=$5

rec_name=$(basename -- $wavpath)
audio_format=(${wavpath##*.})
alt_id=(${rec_name//$(echo ".$audio_format")/ })
lang_dir=data/lang_${rec_id}
output_path=${savepath}

[[ ! -L "steps" ]] && ln -s $KALDI_ROOT/egs/wsj/s5/steps
[[ ! -L "utils" ]] && ln -s $KALDI_ROOT/egs/wsj/s5/utils

echo; echo; echo "===== Starting alignment at  $(date +"%D_%T") ====="; echo

rm -r data
rm -r exp
rm -r $output_path
# rm -r mfcc


# Follow ASA methodology, where we pre-isolate the vocals from the mixture to
# improve vocal-activity detection, resulting in better
# posteriors and alignments a each time point
outdir_ss=$savepath/preisolated_vox
mkdir -p ${outdir_ss}
if [[ $stage -le 0 ]]; then
  echo "PRE-ISOLATION"
  cd demucs
  python3 -m demucs.separate --dl -n demucs -d cpu -o ../${outdir_ss} $wavpath
  cd ..
  mv ${outdir_ss}/demucs/${alt_id}/vocals.wav ${outdir_ss}/${rec_id}_vocals.wav
  rm -r ${outdir_ss}/demucs/${alt_id}/
  rm -r ${outdir_ss}/demucs/
fi


# Use utterance times of syncronized lyrics to remove extraneous non-lyrical
# moments from the separated vocals to prevent alignments in non-vocal regions.
# Keep each song as having one utterance still since the training models were
# created with such pre-conditions
wavpath_vocals=${outdir_ss}/${rec_id}_vocals.wav
if [[ $stage -le 1 ]]; then
    echo "SILENCE NON_UTTERANCES"
    conda activate mirdata
    cp $timed_lyricspath ${outdir_ss}/lyrics.txt
    cp ${outdir_ss}/${rec_id}_vocals.wav ${outdir_ss}/${rec_id}_vocals_raw.wav
    python3 local/process_segmentation_al3625.py ${outdir_ss}/${rec_id}_vocals.wav $timed_lyricspath 0.65 0.2
    conda deactivate
    echo "finished processing segmentation"
fi


# Formatting the non-segmented/non-syncronized version of the lyrics, the
# separated vocals, and the original audio mixture to exist within the Kaldi
# style using the same methodology as the ASA recipe.
if [[ $stage -le 2 ]]; then
    echo "PREP DATA"
    mkdir -p data/${rec_id}
    mkdir -p data/${rec_id}_vocals
    python3 local/data_preparation.py $lyricspath ${outdir_ss}/${rec_id}.wav conf/dict/lexicon_raw.txt data/${rec_id}
    python3 local/data_preparation.py $lyricspath $wavpath_vocals conf/dict/lexicon_raw.txt data/${rec_id}_vocals
    ./utils/fix_data_dir.sh data/${rec_id}
    ./utils/fix_data_dir.sh data/${rec_id}_vocals
fi


# Following ASA recipe, use the grapheme to phoneme phonetisaurus to extend
# the lexicon of the pre-trained model to incorporate any words not already
# in the lexicon allowing for better posterior and lyrical alignmnent.
# Update all lexical/language files to account for this
if [[ $stage -le 3 ]]; then
    echo "PREP LEXICON"
    mkdir -p data/local
    cp -r conf/dict data/local/dict
    ./steps/dict/apply_g2p_phonetisaurus.sh --nbest 2 data/${rec_id}_vocals/oov_words.txt model/g2p data/local/${rec_id}
    cut -d$'\t' -f1,3 data/local/${rec_id}/lexicon.lex > data/local/${rec_id}/lex
    sed -e 's/\t/ /g' data/local/${rec_id}/lex > data/local/${rec_id}/oov_lexicon.txt
    cat data/local/${rec_id}/oov_lexicon.txt data/local/dict/lexicon_raw.txt | sort -u > data/local/dict/lexicon.txt
    sed -e 's/ / 1.0\t/' data/local/dict/lexicon.txt > data/local/dict/lexiconp.txt

    utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang $lang_dir
    silphonelist=$(cat $lang_dir/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang_dir/phones/nonsilence.csl) || exit 1;
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang_dir/topo
    cp -r $lang_dir ${lang_dir}_original
fi


#  Use ASA recipe's lyric segmentation pipeline to create predictions of
#  utterance locations. This proved better than using the hand-annotated
#  utterances as it could be more precise and wasn't biased by inaccurate
#  annotations. However, performance signficantly improved when altering the
#  pre-isolated vocals to silence regions with no true/clear vocals detected
model_dir_chain=model/ctdnnsa_ivec
if [[ $stage -le 4 ]]; then
    echo "SEGMENT AUDIO"
    echo
    ./local/run_lyrics_segmentation.sh --dataset_id ${rec_id}_vocals \
      --wavpath_orig $wavpath --wavpath_vocals $wavpath_vocals \
      --data_orig data/${rec_id} data/${rec_id}_vocals \
      $model_dir_chain $lang_dir || exit 1
fi


# Use modified language model, ivec model, and CTDNN model from ALTA recipe to
# get mfccs, ivecs, and ultimately alignments and posteriorgrams
data_dir_segmented=data/${rec_id}_vocals_vadseg
data_dir_final=data/${rec_id}_vadseg
data_id=$(basename -- $data_dir_segmented)
tree_dir=model/tree
acoustic_model_dir=model/ctdnnsa_ivec
if [[ $stage -le 5 ]]; then
    echo "CREATING ALIGNMENT"
    echo "Extracting features for alignment"
    echo
    utils/fix_data_dir.sh $data_dir_segmented
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 --mfcc-config conf/mfcc_hires.conf \
      $data_dir_segmented exp/make_mfcc/${n}_vadseg mfcc
    steps/compute_cmvn_stats.sh $data_dir_segmented
    utils/fix_data_dir.sh $data_dir_segmented
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 1 \
      ${data_dir_segmented} model/ivector/extractor \
      model/ivector/ivectors_${data_id}_hires

    echo "Force alignment using Phoneme-based CTDNN_SA model from ALTA recipe"
    echo
    ali_dir=exp/${rec_id}_vocals/${rec_id}_vocals_ali
    local/align_chain.sh --cmd "$train_cmd" --nj 1 --beam 100 --retry_beam 7000 \
      --frames_per_chunk 140 --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' --use-gpu false \
      --online_ivector_dir model/ivector/ivectors_${data_id}_hires \
      $data_dir_segmented $lang_dir $model_dir_chain $ali_dir
    ./local/generate_output_alignment.sh --frame_shift 0.03 $data_dir_segmented $rec_id ${lang_dir}_original $ali_dir $savepath
    echo
    echo "Alignment done"

    echo
    echo "CREATING POSTERIORGRAM"
    steps/chain/get_phone_post.sh --remove-word-position-dependency true \
      --online_ivector_dir model/ivector/ivectors_${data_id}_hires \
      $tree_dir $acoustic_model_dir $lang_dir ${data_dir_segmented} exp/phn_post_${rec_id}
    mkdir -p $output_path
    python3 local/reformat_phone_post.py exp/phn_post_${rec_id} $output_path
    echo "Posteriorgram creation done"
fi


echo
echo "==== - ALIGNMENT & POSTERIORGRAM CREATION FINISHED SUCCESSFULLY! - ===="
echo "Output saved at $savepath"

rm -r exp/${rec_id}_vocals/${rec_id}_vocals_segmentation
rm -r data/${rec_id}*
rm -r data/lang_${rec_id}*
rm -r mfcc
