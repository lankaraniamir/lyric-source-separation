import os
import glob
from pathlib import Path
import json
import numpy as np
import nussl
import torch
from nussl.datasets import transforms as nussl_tfm
from common import utils, argbind, viz
import matplotlib.pyplot as plt
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from nussl.separation.base import MaskSeparationBase, DeepMixin, SeparationException
from torch import nn
# from torch.nn.utils import weight_norm
from ignite.engine import Events, Engine, EventEnum
from nussl.ml import SeparationModel
from nussl.ml.networks.modules import (
    Embedding, DualPath, DualPathBlock, STFT, Concatenate,
    LearnedFilterBank, AmplitudeToDB, RecurrentStack,
    MelProjection, BatchNorm, InstanceNorm, ShiftAndScale
)
from sklearn.preprocessing import OneHotEncoder
from nussl import ml

# import warnings
# warnings.filterwarnings('ignore')
utils.logger()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

i_encoder = False
posterior_depth = False
only_audio = False

def get_corpus(folders):
    corpus = set()
    for folder in folders:
        for subfolder in os.listdir(folder):
            subpath = os.path.join(folder, subfolder)
            for file in os.listdir(subpath):
                if file[-9:] == "final.txt":
                    full_path = os.path.join(subpath, file)
                    afile = open(full_path, "r")
                    alignments = []
                    for line in afile.readlines():
                        word = line.split(" ")[1].strip()
                        alignments.append(word)
                    corpus.update(alignments)
    corpus.add("<NULL>")
    corpus = sorted(corpus)

    global i_encoder
    i_encoder = {word:i for i, word in enumerate(corpus)}
    # o_encoder = OneHotEncoder(categories=[corpus], sparse_output=False)
    # o_encoder.fit(np.array(corpus).reshape(-1, 1))
    return corpus, i_encoder

class AlignedData(nussl.datasets.BaseDataset):
    def get_items(self, folder):
        global i_encoder
        global posterior_depth
        global only_audio
        self.i_encoder = i_encoder
        self.posterior_depth = posterior_depth
        self.only_audio = only_audio
        items = []
        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)
            if os.path.isdir(full_path):
                items.append(full_path)
        return items

    def process_item(self, item):
        mix = None
        sources = {}
        metadata = {"temp": "temp"}
        posterior = np.ndarray([])
        for file in os.listdir(item):
            full_path = os.path.join(item, file)
            if file == "vocals.wav":
                sources["vocals"] = nussl.AudioSignal(full_path,
                                                      stft_params=self.stft_params,
                                                      sample_rate=44100)
                if not sources["vocals"]:
                    print(full_path)
                    print("VOCALS NO DATA")
                    exit()
                elif not sources["vocals"].is_mono:
                    print(full_path)
                    print("VOCALS IS NOT MONO")
                    exit()
                elif not sources["vocals"].has_data:
                    print(full_path)
                    print("VOCALS NO DATA")
                    exit()
            elif file == "mixture.wav":
                mix = nussl.AudioSignal(full_path, stft_params=self.stft_params,
                                        sample_rate=44100)
                if not mix.is_mono:
                    print(full_path)
                    print(file, "MIX IS NOT MONO")
                    exit()
                elif not mix.has_data:
                    print(full_path)
                    print("MIX NO DATA")
                    exit()
            elif file == "phone_post.npy":
                pfile = open(full_path, "rb")
                posterior = np.load(pfile)
            elif file[-9:] == "final.txt":
                afile = open(full_path, "r")
                alignments = []
                for line in afile.readlines():
                    time_word = line.split(" ")
                    time_word[0] = float(time_word[0])
                    time_word[1] = time_word[1].strip()
                    alignments.append(time_word)

        if ("vocals" not in sources.keys()
        or not sources["vocals"].has_data
        or not mix.has_data):
            print(full_path)
            if "vocals" not in sources.keys() or not sources["vocals"].has_data:
                print("vox")
            if not mix.has_data:
                print("mix")
            exit()
        sources["non-vocals"] = mix - sources["vocals"]

        j = 0
        extra = 0
        spectrogram = sources["vocals"].stft()
        scale = len(spectrogram[0])/len(posterior)
        remaining = scale
        pgram_list = np.ndarray.tolist(posterior)
        newgrams = [None]*len(spectrogram[1])
        if not len(posterior):
            print("***POSTERIOR IS NONE***")
        for i, sample in enumerate(pgram_list):
            while remaining >= 1:
                if j >= len(newgrams):
                    extra += 1
                else:
                    newgrams[j] = sample.copy()
                remaining -= 1
                j += 1
            remaining = scale + remaining
        if newgrams[-1] == None:
            newgrams[-1] = newgrams[-2]

        newgrams = np.asarray(newgrams)
        newgrams = newgrams[:, :, np.newaxis]
        post = torch.from_numpy(newgrams)

        if self.posterior_depth:
            post_dx = torch.diff(post, n=1, dim=0, prepend=torch.zeros((1,41,1)))
            post_ddx = torch.diff(post, n=2, dim=0, prepend=torch.zeros((2,41,1)))
            post = torch.concatenate((post, post_dx, post_ddx), dim=-2)

        if self.i_encoder :
            alignment_full = ["<NULL>"]*len(spectrogram[1])
            alignment_start = ["<NULL>"]*len(spectrogram[1])
            align_scale = len(spectrogram[0])/mix.signal_duration
            times = [i/align_scale for i, frame in enumerate(alignment_full)]

            # Includes lyric at all positions
            cur_alignment = 0
            old_lyric = "<NULL>"
            cur_lyric = alignments[cur_alignment][1]
            cur_time = float(alignments[cur_alignment][0])
            for i, time in enumerate(times):
                if cur_alignment != len(alignments) and time >= cur_time:
                    alignment_full[i] = cur_lyric
                    cur_alignment += 1
                    old_lyric = cur_lyric
                    if cur_alignment == len(alignments):
                        continue
                    cur_time = float(alignments[cur_alignment][0])
                    cur_lyric = alignments[cur_alignment][1]
                else:
                    alignment_full[i] = old_lyric
            # Includes lyric only at the starting points
            cur_alignment = 0
            old_lyric = "<NULL>"
            cur_lyric = alignments[cur_alignment][1]
            cur_time = float(alignments[cur_alignment][0])
            for i, time in enumerate(times):
                if cur_alignment != len(alignments) and time >= cur_time:
                    alignment_start[i] = cur_lyric
                    cur_alignment += 1
                    if cur_alignment == len(alignments):
                        break
                    cur_time = float(alignments[cur_alignment][0])
                    cur_lyric = alignments[cur_alignment][1]

            alignment_full_encoded = np.asarray([self.i_encoder[word] for word in alignment_full]).reshape(-1, 1)
            alignment_start_encoded = np.asarray([self.i_encoder[word] for word in alignment_full]).reshape(-1, 1)
            lyric_full = alignment_full_encoded[:, :, np.newaxis]
            lyric_start = alignment_start_encoded[:, :, np.newaxis]
            lyric_full = torch.from_numpy(lyric_full)
            lyric_start = torch.from_numpy(lyric_start)
            output = {
                'mix': mix,
                'sources': sources,
                'metadata': metadata,
                'posterior': post,
                'lyric_full': lyric_full,
                # 'lyric_start': lyric_start
            }
        elif self.only_audio:
            output = {
                'mix': mix,
                'sources': sources,
                'metadata': metadata,
            }
        else:
            output = {
                'mix': mix,
                'sources': sources,
                'metadata': metadata,
                'posterior': post
            }

        return output

def get_transforms(keys):
    if not keys:
        keys = []
    return nussl_tfm.Compose([
        nussl_tfm.MagnitudeSpectrumApproximation(),
        nussl_tfm.IndexSources('source_magnitudes', 1),
        nussl_tfm.ToSeparationModel(),
        nussl_tfm.GetExcerpt(8192, tf_keys=(['mix_magnitude', 'source_magnitudes', 'ideal_binary_mask', 'weights'] + keys))
    ])

def get_data(ref_path, keys, post_depth=False, use_corpus=False,
             stft_params=nussl.STFTParams(window_length=512, hop_length=128), audio_only=False):
    main_folder = os.path.join(ref_path,"../1_create_alignments/results")
    full_train_folder = os.path.join(main_folder, "full_train")
    val_folder = os.path.join(main_folder, "dev")
    test_folder = os.path.join(main_folder, "test")

    global posterior_depth
    posterior_depth = post_depth

    if use_corpus:
        corpus, _ = get_corpus([full_train_folder, val_folder, test_folder])
    else:
        global i_encoder
        i_encoder = False

    global only_audio
    only_audio = audio_only

    rel_transform = get_transforms(keys)

    train_data = AlignedData(full_train_folder,
                             transform=rel_transform, stft_params=stft_params,
                             sample_rate=44100)
    val_data = AlignedData(folder=val_folder,
                            transform=rel_transform, stft_params=stft_params,
                            sample_rate=44100)
    test_data = AlignedData(folder=test_folder,
                            transform=None, stft_params=stft_params,
                            sample_rate=44100)

    if use_corpus:
        return train_data, val_data, test_data, corpus
    return train_data, val_data, test_data


class ValidationEvents(EventEnum):
    VALIDATION_STARTED = 'validation_started'
    VALIDATION_COMPLETED = 'validation_completed'
class BackwardsEvents(EventEnum):
    BACKWARDS_COMPLETED = 'backwards_completed'
def modified_create_train_and_validation_engines(train_func, val_func=None, device='cpu'):
    trainer = Engine(train_func)
    trainer.register_events(*ValidationEvents)
    trainer.register_events(*BackwardsEvents)
    validator = None if val_func is None else Engine(val_func)
    device = device if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    def prepare_batch(engine):
        batch = engine.state.batch
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].float().to(device)
        engine.state.batch = batch
    def book_keeping(engine):
        if "epoch_history" not in dir(trainer.state):
            engine.state.epoch_history = {}
        engine.state.iter_history = {}
        engine.state.past_iter_history = {}
    def add_to_iter_history(engine):
        for key in engine.state.output:
            if key not in engine.state.iter_history:
                engine.state.iter_history[key] = []
            if key not in engine.state.past_iter_history:
                engine.state.past_iter_history[key] = []
            engine.state.iter_history[key].append(
                engine.state.output[key]
            )
            engine.state.past_iter_history[key].append(
                engine.state.iter_history[key]
            )
    def clear_iter_history(engine):
        engine.state.iter_history = {}
    trainer.add_event_handler(
        Events.ITERATION_STARTED, prepare_batch)
    trainer.add_event_handler(
        Events.STARTED, book_keeping)
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, add_to_iter_history)
    trainer.add_event_handler(
        Events.EPOCH_STARTED, clear_iter_history)
    if validator is not None:
        validator.add_event_handler(
            Events.ITERATION_STARTED, prepare_batch)
        validator.add_event_handler(
            Events.STARTED, book_keeping)
        validator.add_event_handler(
            Events.ITERATION_COMPLETED, add_to_iter_history)
        validator.add_event_handler(
            Events.EPOCH_STARTED, clear_iter_history)
    return trainer, validator

def convert_mono(dir):
    import os
    from pydub import AudioSegment
    for folder in os.listdir(dir):
        for file in os.listdir(folder):
                if (file[-3:] == "wav"):
                        full_path = os.path.join(folder, file)
                        sound = AudioSegment.from_wav(full_path)
                        sound = sound.set_channels(1)
                        sound.export(full_path, format="wav")


class DeepMaskEstimationPosterior(DeepMixin, MaskSeparationBase):
    def __init__(self, input_audio_signal, posterior, model_path=None, device='cpu',
                 **kwargs):
        super().__init__(input_audio_signal, **kwargs)
        if model_path is not None:
            self.load_model(model_path, device=device)
        self.model_output = None
        self.posterior = posterior
        self.channel_dim = -1

    def forward(self, **kwargs):
        input_data = self._get_input_data_for_model(**kwargs)
        input_data['posterior'] = self.posterior.unsqueeze(0).to(torch.float32).to(DEVICE)
        with torch.no_grad():
            output = self.model(input_data)
            if 'mask' not in output:
                raise SeparationException(
                    "This model is not a deep mask estimation model! "
                    "Did not find 'mask' key in output dictionary.")
            masks = output['mask']
            if self.metadata['num_channels'] == 1:
                masks = masks.transpose(0, -2)
            masks = masks.squeeze(0).transpose(0, 1)
            masks = masks.cpu().data.numpy()
        self.model_output = output
        return masks

    def run(self, masks=None):
        self.result_masks = []
        if masks is None:
            masks = self.forward()
        for i in range(masks.shape[-1]):
            mask_data = masks[..., i]
            if self.mask_type == self.MASKS['binary']:
                mask_data = masks[..., i] == masks.max(axis=-1)
            mask = self.mask_type(mask_data)
            self.result_masks.append(mask)
        return self.result_masks

    def confidence(self, approach='silhouette_confidence', num_sources=2, **kwargs):
        if self.model_output is None:
            raise SeparationException(
                "self.model_output is None! Did you run forward?")
        if 'embedding' not in self.model_output:
            raise SeparationException(
                "embedding not in self.model_output! Can't compute confidence.")
        features = self.model_output['embedding']
        if self.metadata['num_channels'] == 1:
            features = features.transpose(0, -2)
        features = features.squeeze(0).transpose(0, 1)
        features = features.cpu().data.numpy()

        confidence_function = getattr(ml.confidence, approach)
        confidence = confidence_function(
            self.audio_signal, features, num_sources, **kwargs)
        return confidence

class DeepMaskEstimationAlign(DeepMixin, MaskSeparationBase):
    def __init__(self, input_audio_signal, posterior, lyric_full, model_path=None, device='cpu',
                 **kwargs):
        super().__init__(input_audio_signal, **kwargs)
        if model_path is not None:
            self.load_model(model_path, device=device)
        self.model_output = None
        self.posterior = posterior
        self.lyric_full = lyric_full
        self.channel_dim = -1

    def forward(self, **kwargs):
        input_data = self._get_input_data_for_model(**kwargs)
        input_data['posterior'] = self.posterior.unsqueeze(0).to(torch.float32).to(DEVICE)
        input_data['lyric_full'] = self.lyric_full.unsqueeze(0).long().to(DEVICE)
        with torch.no_grad():
            output = self.model(input_data)
            if 'mask' not in output:
                raise SeparationException(
                    "This model is not a deep mask estimation model! "
                    "Did not find 'mask' key in output dictionary.")
            masks = output['mask']
            if self.metadata['num_channels'] == 1:
                masks = masks.transpose(0, -2)
            masks = masks.squeeze(0).transpose(0, 1)
            masks = masks.cpu().data.numpy()
        self.model_output = output
        return masks

    def run(self, masks=None):
        self.result_masks = []
        if masks is None:
            masks = self.forward()
        for i in range(masks.shape[-1]):
            mask_data = masks[..., i]
            if self.mask_type == self.MASKS['binary']:
                mask_data = masks[..., i] == masks.max(axis=-1)
            mask = self.mask_type(mask_data)
            self.result_masks.append(mask)
        return self.result_masks

    def confidence(self, approach='silhouette_confidence', num_sources=2, **kwargs):
        if self.model_output is None:
            raise SeparationException(
                "self.model_output is None! Did you run forward?")
        if 'embedding' not in self.model_output:
            raise SeparationException(
                "embedding not in self.model_output! Can't compute confidence.")
        features = self.model_output['embedding']
        if self.metadata['num_channels'] == 1:
            features = features.transpose(0, -2)
        features = features.squeeze(0).transpose(0, 1)
        features = features.cpu().data.numpy()
        confidence_function = getattr(ml.confidence, approach)
        confidence = confidence_function(
            self.audio_signal, features, num_sources, **kwargs)
        return confidence


def visualize_masks(sources, y_axis='mel'):
    plt.figure(figsize=(10, 4))
    plt.subplot(111)
    nussl.utils.visualize_sources_as_masks(
        sources, db_cutoff=-61, y_axis=y_axis)
    ax = plt.gca()
    ax.get_legend().remove()
    plt.tight_layout()
    plt.show()


def embed_audio(sources):
    nussl.play_utils.multitrack(sources, ext='.wav')