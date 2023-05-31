# Lyric Source Separation
Code for research paper ["Using Synchronized Lyrics as a Novel Input into
Vocal Isolation Models"](https://github.com/lankaraniamir/lyric_source_separation/blob/main/Using%20Synchronized%20Lyrics%20as%20a%20Novel%20Input%20into%20Vocal%20Isolation%20Models.pdf)


## Project Summary
Source separation projects have typically only focused on using audio data
as an input into the separation network, so this project tests whether or not
other external information about the music can be used as a method to bolster
this separation. More specifically, I took the standard Musdb source separation
dataset, alongside annotations of all of its lyrics and time points, and hoped
to test whether these lyrics could help the model learn certain phonetic traits
that would make it easier to isolate what parts of the frequency spectrum are
associated with a vocalist. To create a more expansive dataset, I altered
Demucs' audio remixing program to adapt the speed, pitch, and texture of the
input audio around the vocals while only splitting based upon the start of a
lyrical phrase to keep text position estimations in tact. Then this was fed into
a modified combination of Emir Demirel's ASA & ALTA Kaldi recipes predicting the
exact moment where a word is spoken to create alignment text files, as well as
predicting the chance each phoneme is said at every frame of audio in order to
create a posteriorgram. These features were then fed into a Pytorch source
separation network based upon the NUSSL package to see if it could improve our
separation. Posteriorgrams were concatenated onto the audio before the
RNN chain and, when used, the lyrics were encoded and embedded using linear
models which were then matched with the RNN at the final layer. These models
were based upon finding a spectral mask for the vocal audio in hopes that this
would map well with the posteriorgrams' dimensionality. Sadly, it seems that the
separation showed little to no improvements with the posteriorgrams and the
alignments hindered training as it overly complicated and misled the network
biasing it towards a few words. To improve: a significnatly larger amount of
data needs to be procured with lyrics and stems; a more nuanced integration of
the audio features in our model is needed; and there should probably
be posteriorgrams and alignments that are predicted at the same frame
rate as the spectrogram for a consistent correlation between the two.


## List of Tools Needed
- [Demucs](https://github.com/facebookresearch/demucs) to run modified version of [Demucs' automixing program](https://github.com/facebookresearch/demucs/blob/main/tools/automix.py)
  as well as to do the initial source separation used to create more
  accurate posteriograms and alignments in Kaldi.
  
- [ALTA recipe](https://github.com/emirdemirel/ALTA) for Kaldi was used to create the models for alignment and
posteriogram creation and was also used for the actual creation of the
posteriorgrams.

- [ASA recipe](https://github.com/emirdemirel/ASA_ICASSP2021) for Kaldi was used to create the alignments. Quite similar to
ALTA so I was able to use the same models for each

- [Nussl](https://github.com/nussl/nussl
) was used as a general framework for source separation to make a
good starting point to compare posteriogram addition to. Further additions
were made using Pytorch.

- [High quality version of the musdb dataset](https://zenodo.org/record/3338373) standard for source separation

- [Annotations of the lyrics](https://zenodo.org/record/3989267) in each song of the Musdb dataset


## Directories and executables

Part 0: Create Conda environments
- Build each conda environment from the yml files using:

    conda env create -f ASA.yml
    <br />conda env create -f demucs.yml
    <br />conda env create -f mirdata.yml
    <br />conda env create -f sep.yml

- In part 1, the first three environments will be activate and deactivated
automatically
- For part 2, activate the sep environment for all testing
- Due to the usage of varying packages of different ages, different
conda environments were the only way I could get things to work
portably.


<br />Part 1: Create alignments
- This directory contains all of the code to bolster the dataset and to
create alignments/posteriograms.
- run_al3625.sh - main script
- align_and_post_al3625.sh - this is the script to extract the
alignments & posteriorgram from any one file using the models trained
from the ALTA recipe. This works by first isolating the vocals from the
audio using a very basic pre-trained source separation model meant to
be useful for this exact purpose. Then we silence moments where no
lyrics are being spoken to create better alignments and posteriorgrams.
Then we use the ivecs & mfccs to get the our output as described in the
file itself. Most of the modifications come in terms of getting the
desired outputon a larger scale and making the two be extracted by
similar means.
- local/data_preparation... - These files are used to process the Musdb
audio and its lyrics to create output files in Kaldi format as well as in
ALTA/ASA format creating the general tagging and directory structure.
- posteriorgram_visual_al3625.ipynb - simple visual of a posteriorgram
for reference and understanding (better one in next section)
- local/mashup_with... - creates the remixed version of the songs as
well as the new aligned lyrics of this output by shifting chromas and
tempos of a set and then choosing a random match of these features that
satisfies our minimum criteria. Most of my modifications revolved
around creating modifications at lyric breakpoints that resulted in an
outputting of correctly aligned lyrics, modifying the code be more
relativistic to the vocals, and extending the range of variance to
account for our smaller dataset since we can only use transcribed audio.
- local/prep_data... - preps the processed data to create lexical data
for Kaldi
- local/process_segmentation_al3625.py - fades out parts of the
pre-isolated vocals that should not have audio based on the lyric files
to create cleaner and more accurate posteriorgrams and alignments


<br />Part 2: Separate Sources
- This directory uses the Kaldi input to do the actual source separation
- setup_al3625.py - defines most of the groundwork to how all the models are
similarly designed. The most notable part of this is the aligned data audio class
which processes posteriorgrams to exist on the same scale as the audio
file and creates a vector of the current lyric encoding at each audio frame.
- training_scripts - these were used to define and train each of the
distinct models. They also were used to evaluate the models after
training (although I didn't make time to do a final evaluation for a few
of the less important models). Finally, at the end you can visualize the
spectral mask for any evaluated model as well as listen to its
prediction of what is and isn't the vocal. The basic setup follow the
Nussl backbone closely so that the model we compare to is compared to is
a standard baseline. However, lots of modifications were made to support
our specific data and make it interactable with non-audio data.
    - audio_post_no_norm: model using the audio and an unnormalized
    posterirogram. This in general has the most evaluation tools as I
    used it to create the figures for the paper. If would like similar
    images of other model simply copy the codes for that section. This
    includes an image of predicted real mask and an image of what
    the posteriorgram and spectrogram concatenation looks like
    - all_audio: basic source separation model with no additions
    - all_post: model without any audio data and just using
    posteriorgrams
    - audio_post_aligns: model using audio data, posteriorgrams, and the
    lyrical alignments
    - audio_post_no_norm_larger: model using the audio and an unnormalized
    posterirogram with extra posteriorgram rows (as dx & ddx of each row
    of posteriorgram) and larger hidden_layers
    - audio_post_norm: model using the audio and a normalized posteriogram
- eval_results: contain jsons of each eval test
- output_audio: exported audio from evaluation
- trained_models: contain the newest and best trained models &
optimizers of each type
- compare_training_al3625.ipynb - simple visual comparing loss of models
- get_eval_scores_al3625.ipynb - this will take the scores from the
evaluation of the decoded models and output them into clean tables for
all of the important evaluation features


## How to run
- Download all tools neeeded
- Download databases and place into db folder in 1_create_alignments
- Train all models needed from trained github project and put into model in
1_create_alignments
- Run the part 0 code to setup conda environments as described above
- Run the part 1 file: run_al3625.sh. This will first prep the musdb data,
 create remixes, prep the mashup data, and then prep the mashup data. Then 
it will go through each of the mono audio files and feed it into 
align_and_post_al3625 to get the alignments and posteriograms for that song
- Run the part 2 training_scripts - Each are simple ipynb notebooks that can 
simply be gone through in any way you like. If you would like to hear any 
output audio and see the output masks, run the visualization functions commented
out of the code
