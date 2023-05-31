# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This is a modification of the automix function from demucs optimizing it to
# extract only at specific beat onsets that roughly correspond with the start of
# a new utterance in the original files instead of chopping at random
"""
This is an adaptation
This script creates realistic mixes with stems from different songs.
In particular, it will align BPM, sync up the first beat and perform pitch
shift to maximize pitches overlap.
In order to limit artifacts, only parts that can be mixed with less than 15%
tempo shift, and 3 semitones of pitch shift are mixed together.
"""
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import random
import shutil
import tqdm
import pickle

from librosa.beat import beat_track
from librosa.feature import chroma_cqt
import numpy as np
import torch
from torch.nn import functional as F

from dora.utils import try_load
from demucs.audio import save_audio
from demucs.repitch import repitch
from demucs.pretrained import SOURCES
from demucs.wav import build_metadata, Wavset, _get_musdb_valid, _track_metadata

import pandas as pd

COUNT=0

musdb_data = pd.read_csv("db/musdb18hq/musdb_hq_all_data.csv")

MUSDB_PATH = "db/other/musdb18hq_backup"
EXTRA_WAV_PATH = "db/my_stems"
OUTPATH = "db/mashups_mono/"
CACHE = "demucs/automix_cache/"

CHANNELS = 1
SR = 44100
MAX_PITCH = 4     # prev was 3
MAX_TEMPO = 0.25  # prev was .15

# Added lyrics & lyric positions
# Old: Spec = namedtuple("Spec", "tempo onsets kr track index ")
Spec = namedtuple("Spec", "tempo onsets kr track index lyrics lyric_times")

def rms(wav, window=10000):
    """efficient rms computed for each time step over a given window."""
    half = window // 2
    window = 2 * half + 1
    wav = F.pad(wav, (half, half))
    tot = wav.pow(2).cumsum(dim=-1)
    return ((tot[..., window - 1:] - tot[..., :-window + 1]) / window).sqrt()

def analyse_track(dset, index):
    """analyse track, extract bpm and distribution of notes from the bass line."""
    track = dset[index]
    mix = track.sum(0).mean(0)
    ref = mix.std()

    starts = (abs(mix) >= 1e-2 * ref).float().argmax().item()
    track = track[..., starts:]

    cache = CACHE / dset.sig
    cache.mkdir(exist_ok=True, parents=True)

    cache_file = cache / f"{index}.pkl"
    cached = None
    if cache_file.exists():
        cached = try_load(cache_file)
        if cached is not None:
            tempo, events, hist_kr = cached

    if cached is None:
        drums = track[0].mean(0)
        if drums.std() > 1e-2 * ref:
            tempo, events = beat_track(drums.numpy(), units='time', sr=SR)
        else:
            print("failed drums", drums.std(), ref)
            return None, track

        bass = track[1].mean(0)
        r = rms(bass)
        peak = r.max()
        mask = r >= 0.05 * peak
        bass = bass[mask]
        if bass.std() > 1e-2 * ref:
            kr = torch.from_numpy(chroma_cqt(bass.numpy(), sr=SR))
            hist_kr = (kr.max(dim=0, keepdim=True)[0] == kr).float().mean(1)
        else:
            print("failed bass", bass.std(), ref)
            return None, track

    # Added
    name = list(dset.metadata.keys())[index]
    artist, song = name.split(" - ")
    relevant_lines = musdb_data.loc[(musdb_data["artist"] == artist) & (musdb_data["song"] == song)]
    lyrics = relevant_lines.lyrics.values
    lyric_times = relevant_lines.start.values

    pickle.dump([tempo, events, hist_kr], open(cache_file, 'wb'))
    spec = Spec(tempo, events, hist_kr, track, index, lyrics, lyric_times)
    # OLD: spec = Spec(tempo, events, hist_kr, track, index)
    return spec, None

def best_pitch_shift(kr_a, kr_b):
    """find the best pitch shift between two chroma distributions."""
    deltas = []
    for p in range(12):
        deltas.append((kr_a - kr_b).abs().mean())
        kr_b = kr_b.roll(1, 0)

    ps = np.argmin(deltas)
    if ps > 6:
        ps = ps - 12
    return ps

def align_stems(stems, lyrics, lyric_times):
    """Align the first beats of the stems.
    This is a naive implementation. A grid with a time definition 10ms is defined and
    each beat onset is represented as a gaussian over this grid.
    Then, we try each possible time shift to make two grids align the best.
    We repeat for all sources.
    """
    sources = len(stems)
    width = 5e-3  # grid of 10ms
    limit = 5
    std = 2
    x = torch.arange(-limit, limit + 1, 1).float()
    gauss = torch.exp(-x**2 / (2 * std**2))

    # print(stems[3][1][-15:], true_times[-15:], true_lyrics[-15:])
    # print(stems[3][1][:], true_times[:])#, true_lyrics[:])

    grids = []
    for wav, onsets in stems:
        le = wav.shape[-1]
        dur = le / SR
        grid = torch.zeros(int(le / width / SR))
        for onset in onsets:
            pos = int(onset / width)
            if onset >= dur - 1:
                continue
            if onset < 1:
                continue
            grid[pos - limit:pos + limit + 1] += gauss
        grids.append(grid)

    # print(stems[3][1][:10], onset_times[:10], onset_lyrics[:10])

    shifts = [0]
    for s in range(1, sources):
        max_shift = int(4 / width)
        dots = []
        for shift in range(-max_shift, max_shift):
            other = grids[s]
            ref = grids[0]
            if shift >= 0:
                other = other[shift:]
            else:
                ref = ref[shift:]
            le = min(len(other), len(ref))
            dots.append((ref[:le].dot(other[:le]), int(shift * width * SR)))

        _, shift = max(dots)
        shifts.append(-shift)

    outs = []
    new_zero = min(shifts)

    for (wav, _), shift in zip(stems[:3], shifts[:3]):
        offset = shift - new_zero
        wav = F.pad(wav, (offset, 0))
        outs.append(wav)


    # shifted_times = [v_onset + v_shift/1000]
    # print(shifted_times)
    # shifted_times[start_index] = max(shifted_times[start_index], 0)

    # print("\n\n")
    # print("\nPREDICTED ONSETS : ", [round(o, 3) for o in pred_times])
    # print("\nTRUE ONSETS: ", true_times)

    v_wav, onsets = stems[3]
    v_shift = shifts[3]

    v_offset = v_shift - new_zero
    v_wav = F.pad(v_wav, (v_offset, 0))
    outs.append(v_wav)

    le = min(x.shape[-1] for x in outs)
    v_le = outs[3].shape[-1]
    # print(le/SR, v_le/SR, v_offset/SR, "vocal_length is mininimum: ", v_le == le)
    # print(round(v_offset, 3), round(v_shift, 3))
    # print(v_le, outs[2].shape[-1], outs[1].shape[-1], outs[0].shape[-1])
    # updated_times = [time-v_shift/SR for time in lyric_times]

    # updated_times = [time+v_offset/SR for time in lyric_times]
    updated_times = [round(time+v_offset/SR, 2) for time in lyric_times]
    if le < v_le:
        for time in updated_times[::-1]:
            if time < le/SR:
                le = round(time*SR)
                break
        else:
            print("\n*****ERRRORRRRRRR AMIR*****")
            updated_times = [0]
            lyrics = ["<NO_VOX>"]
            print(updated_times, le/SR, v_le/SR)


    outs = [w[..., :le] for w in outs]
    audio_length = round(le / SR, 2)
    final_times = [time for time in updated_times if time<audio_length]
    final_lyrics = [lyrics[i] for i, time in enumerate(updated_times) if time<audio_length]
    return torch.stack(outs), final_lyrics, final_times, audio_length

    # return torch.stack(outs), lyrics, lyric_times, audio_length

    # return torch.stack(outs), best_lyrics, best_times, audio_length
    # print(audio_length)
    # print(stems[3][1], best_times)

    # return torch.stack(outs), final_lyrics, final_times, audio_length
    # return torch.stack(outs), best_lyrics, best_times, audio_length

    # best_times = []
    # best_lyrics = []
    # # cur_bound = pred_onsets[0]
    # if len(true_times) != 0:
    #     cur_bound = true_times[0]
    #     min_diff = 1
    #     min_index = 0
    #     i = 0
    #     for pred_time, true_time in zip(pred_times, true_times):
    #         cur_diff = abs(true_time-pred_time)
    #         if cur_bound != true_time:
    #             best_times.append(pred_times[min_index])
    #             best_lyrics.append(true_lyrics[min_index])
    #             cur_bound = true_times[i]
    #             min_diff = cur_diff
    #             min_index = i
    #         else:
    #             if cur_diff < min_diff:
    #                 min_diff = cur_diff
    #                 min_index = i
    #             if i == len(true_times)-1:
    #                 best_times.append(pred_times[min_index])
    #                 best_lyrics.append(true_lyrics[min_index])
    #         i += 1

    # shifted_times = [onset + v_shift/1000) for onset in best_times]
    # start_index = shifted_times.index(min(shifted_times))

    # shifted_times1 = [(-onset + offset/1000) for onset in best_times]
    # # shifted_times2 = [abs(onset + v_shift/1000) for onset in best_times]
    # start_index = shifted_times2.index(min(shifted_times2))
    # shifted_times3 = [(-onset + offset*width) for onset in best_times]

    # test_times =   shifted_times2[start_index:]
    # test_lyrics =   best_lyrics[start_index:]


    # print("TRUE ONSETS: ", true_times)
    # print("SHIFTED ONSETS1: ", shifted_times1)
    # print("SHIFTED ONSETS2: ", shifted_times2)
    # print("SHIFTED LYRICS2: ", shifted_times2)
    # print("\nREMAINING TIMES: ", test_times)
    # print("REMAINING LYRICS: ", test_lyrics)
    # print("SHIFTED ONSETS3: ", shifted_times3)
    # print("*** *** *** ", v_offset, " *** *** ***")

def find_candidate(spec_ref, catalog, pitch_match=True):
    """Given reference track, this finds a track in the catalog that
    is a potential match (pitch and tempo delta must be within the allowable limits).
    """
    candidates = list(catalog)
    random.shuffle(candidates)

    for spec in candidates:
        ok = False
        for scale in [1/4, 1/2, 1, 2, 4]:
            tempo = spec.tempo * scale
            delta_tempo = spec_ref.tempo / tempo - 1
            if abs(delta_tempo) < MAX_TEMPO:
                ok = True
                break
        if not ok:
            print(delta_tempo, spec_ref.tempo, spec.tempo, "FAILED TEMPO")
            # too much of a tempo difference
            continue
        spec = spec._replace(tempo=tempo)

        ps = 0
        if pitch_match:
            ps = best_pitch_shift(spec_ref.kr, spec.kr)
            if abs(ps) > MAX_PITCH:
                print("Failed pitch", ps)
                # too much pitch difference
                continue
        return spec, delta_tempo, ps

def get_part(spec, source, dt, dp):
    """Apply given delta of tempo and delta of pitch to a stem."""
    wav = spec.track[source]
    if dt or dp:
        wav = repitch(wav, dp, dt * 100, samplerate=SR, voice=source == 3)
        spec = spec._replace(onsets=spec.onsets / (1 + dt))
    return wav, spec

def get_vox_part(spec, source, dt, dp):
    """Apply given delta of tempo and delta of pitch to a stem."""
    wav = spec.track[source]
    if dt or dp:
        wav = repitch(wav, dp, dt * 100, samplerate=SR, voice=source == 3)
        spec = spec._replace(onsets=spec.onsets / (1 + dt))
        spec = spec._replace(lyric_times=spec.lyric_times / (1 + dt))
    return wav, spec

def build_track(ref_index, catalog):
    """Given the reference track index and a catalog of track, builds
    a completely new track. One of the source at random from the ref track will
    be kept and other sources will be drawn from the catalog.
    """
    # Index 3 = vocals if not randomized so avoid to isolate vox
    order = list(range(len(SOURCES)))
    random.shuffle(order)
    # order.reverse()

    stems = [None] * len(order)
    indexes = [None] * len(order)
    origs = [None] * len(order)
    dps = [None] * len(order)
    dts = [None] * len(order)

    # First track is now always vox track here
    # first = order[3]
    first = order[0]
    spec_ref = catalog[ref_index]

    # # Added but temp removed
    # # print("\n*** ONSETS & LYRIC TIMES ***")
    # # print("PRIOR: ", spec_ref.onsets[:10], spec_ref.lyric_times, onset_lyrics[:10])
    # # My code - Eliminating onsets not at beat start
    # # onset_lyrics = []
    # # onset_times = []
    # deleted = 0
    # temp_onsets = spec_ref.onsets.copy()
    # temp_lyric_times = spec_ref.lyric_times.copy()
    # temp_lyrics = spec_ref.lyrics.copy()
    # for onset_index, onset in enumerate(temp_onsets):
    #     for lyr_start in temp_lyric_times:
    #         diff = onset - lyr_start
    #         if abs(diff) < .5:
    #             break
    #             # print(onset, lyr_start, diff)
    #             # onset_lyrics.append(spec_ref.lyrics[lyr_index])
    #             # onset_times.append(spec_ref.lyric_times[lyr_index])
    #     else:
    #         spec_ref = spec_ref._replace(onsets = np.delete(spec_ref.onsets, onset_index-deleted))
    #         deleted += 1
    #         continue

    stems[first] = (spec_ref.track[first], spec_ref.onsets)
    indexes[first] = ref_index
    origs[first] = spec_ref.track[first]
    dps[first] = 0
    dts[first] = 0

    pitch_match = order != 0

    times = []
    # for src in order[:3]:
    for src in order[1:]:
        spec, dt, dp = find_candidate(spec_ref, catalog, pitch_match=pitch_match)
        if not pitch_match:
            spec_ref = spec_ref._replace(kr=spec.kr)
        pitch_match = True
        dps[src] = dp
        dts[src] = dt
        if src == 3:
            wav, spec = get_vox_part(spec, src, dt, dp)
            times = spec.lyric_times
            lyrics = spec.lyrics
        else:
            wav, spec = get_part(spec, src, dt, dp)
        stems[src] = (wav, spec.onsets)
        indexes[src] = spec.index
        origs.append(spec.track[src])
    if len(times) == 0:
        times = spec_ref.lyric_times
        lyrics = spec_ref.lyrics

    print("FINAL CHOICES", ref_index, indexes, dps, dts)
    stems, final_lyrics, final_times, audio_length = align_stems(stems, lyrics, times)
    return stems, origs, final_lyrics, final_times, audio_length

def get_musdb_dataset(part='train'):
    root = Path(MUSDB_PATH) / part
    ext = '.wav'
    metadata = build_metadata(root, SOURCES, ext=ext, normalize=False)
    valid_tracks = _get_musdb_valid()
    metadata_train = {name: meta for name, meta in metadata.items() if name not in valid_tracks}
    train_set = Wavset(
        root, metadata_train, SOURCES, samplerate=SR, channels=CHANNELS,
        normalize=False, ext=ext)
    sig = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    train_set.sig = sig
    return train_set

def get_wav_dataset():
    root = Path(EXTRA_WAV_PATH)
    ext = '.wav'
    metadata = _build_metadata(root, SOURCES, ext=ext, normalize=False)
    train_set = Wavset(
        root, metadata, SOURCES, samplerate=SR, channels=CHANNELS,
        normalize=False, ext=ext)
    sig = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    train_set.sig = sig
    return train_set

def main():
    random.seed(4321)
    if OUTPATH.exists():
        shutil.rmtree(OUTPATH)
    OUTPATH.mkdir(exist_ok=True, parents=True)
    (OUTPATH / 'train').mkdir(exist_ok=True, parents=True)
    (OUTPATH / 'valid').mkdir(exist_ok=True, parents=True)
    out = OUTPATH / 'train'
    dset = get_musdb_dataset()
    # dset2 = get_wav_dataset()
    # dset3 = get_musdb_dataset('test')
    dset2 = None
    dset3 = None
    pendings = []
    copies = 15
    copies_rej = 4

    with ProcessPoolExecutor(20) as pool:
        for index in range(len(dset)):
            pendings.append(pool.submit(analyse_track, dset, index))

        if dset2:
            for index in range(len(dset2)):
                pendings.append(pool.submit(analyse_track, dset2, index))
        if dset3:
            for index in range(len(dset3)):
                pendings.append(pool.submit(analyse_track, dset3, index))

        catalog = []
        rej = 0
        for pending in tqdm.tqdm(pendings, ncols=120):
            spec, track = pending.result()
            if spec is not None:
                catalog.append(spec)
            else:
                mix = track.sum(0)
                for copy in range(copies_rej):
                    folder = out / f'rej_{rej}_{copy}'
                    folder.mkdir()
                    save_audio(mix, folder / "mixture.wav", SR)
                    for stem, source in zip(track, SOURCES):
                        if source == "vocals":
                            save_audio(stem, folder / f"{source}.wav", SR, clip='clamp')
                        # save_audio(stem, folder / f"{source}.wav", SR, clip='clamp')
                    rej += 1

    for copy in range(copies):
        count = 0
        for index in range(len(catalog)):
            print(" ")
            print(count)
            track, origs, lyrics, times, audio_length = build_track(index, catalog)
            mix = track.sum(0)
            mx = mix.abs().max()
            scale = max(1, 1.01 * mx)
            mix = mix / scale
            track = track / scale
            folder = out / f'{copy}_{index}'
            folder.mkdir()
            save_audio(mix, folder / "mixture.wav", SR)
            for stem, source, orig in zip(track, SOURCES, origs):
                # save_audio(stem, folder / f"{source}.wav", SR, clip='clamp')
                if source == "vocals":
                    save_audio(stem, folder / f"{source}.wav", SR, clip='clamp')
                # save_audio(stem.std() * orig / (1e-6 + orig.std()), folder / f"{source}_orig.wav",
                #            SR, clip='clamp')

            i = 0
            f = open(folder / "lyrics.txt", "w")
            for i, start in enumerate(times):
                utt = lyrics[i]
                if i + 1 == len(times):
                    stop = audio_length
                else:
                    stop = times[i+1]
                start_str = "{:07.2f}".format(start)
                stop_str = "{:07.2f}".format(stop)
                f.write(f"{start_str} {stop_str} {utt}\n")
            count += 1

if __name__ == '__main__':
    main()
