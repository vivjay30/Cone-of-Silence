import os
import sys

import argparse
import json
from typing import List
from pathlib import Path
import tqdm
import random

import multiprocessing.dummy as mp

import numpy as np
import librosa
import pyroomacoustics as pra
import soundfile as sf

# Mean and STD of the signal peak
FG_TARGET_VOL = 0.25
FG_VOL_STD = 0.15

BG_TARGET_VOL = 0.2
BG_VOL_STD = 0.2


def generate_mic_array(room, mic_radius: float, n_mics: int):
    """
    Generate a list of Microphone objects

    Radius = 50th percentile of men Bitragion breadth
    (https://en.wikipedia.org/wiki/Human_head)
    """
    R = pra.circular_2D_array(center=[0., 0.], M=n_mics, phi0=0, radius=mic_radius)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))


def handle_error(e):
    print(e)


def get_voices(args):
    # Make sure we dont get an empty sequence
    success = False
    while not success:
        voice_files = random.sample(args.all_voices, args.n_voices)
        # Save the identity also. This is VCTK specific
        success = True
        voices_data = []
        for voice_file in voice_files:
            voice_identity = str(voice_file).split("/")[-1].split("_")[0]
            voice, _ = librosa.core.load(voice_file, sr=args.sr, mono=True)
            voice, _ = librosa.effects.trim(voice)
            if voice.std() == 0:
                success = False
            voices_data.append((voice, voice_identity))

    return voices_data


def generate_sample(args: argparse.Namespace, bg: np.ndarray, idx: int) -> int:
    """
    Generate a single sample. Return 0 on success.

    Steps:
    - [1] Load voice
    - [2] Sample background with the same length as voice.
    - [3] Pick background location
    - [4] Create a scene
    - [5] Render sound
    - [6] Save metadata
    """
    # [1] load voice
    output_prefix_dir = os.path.join(args.output_path, '{:05d}'.format(idx))
    Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)

    voices_data = get_voices(args)

    # [2]
    total_samples = int(args.duration * args.sr)
    if bg is not None:
        bg_length = len(bg)
        bg_start_idx = np.random.randint(bg_length - total_samples)
        sample_bg = bg[bg_start_idx:bg_start_idx + total_samples]

    # Generate room parameters, each scene has a random room and absorption
    left_wall = np.random.uniform(low=-20, high=-15)
    right_wall = np.random.uniform(low=15, high=20)
    top_wall = np.random.uniform(low=15, high=20)
    bottom_wall = np.random.uniform(low=-20, high=-15)
    absorption = np.random.uniform(low=0.1, high=0.99)
    corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
                    [   right_wall, top_wall], [right_wall, bottom_wall]]).T

    # FG
    all_fg_signals = []
    voice_positions = []
    for voice_idx in range(args.n_voices):
        # Need to re-generate room to save GT. Could probably be optimized
        room = pra.Room.from_corners(corners,
                                     fs=args.sr,
                                     max_order=10,
                                     absorption=absorption)
        mic_array = generate_mic_array(room, args.mic_radius, args.n_mics)

        voice_radius = np.random.uniform(low=1.0, high=5.0)
        voice_theta = np.random.uniform(low=0, high=2 * np.pi)
        voice_loc = [
            voice_radius * np.cos(voice_theta),
            voice_radius * np.sin(voice_theta)
        ]
        voice_positions.append(voice_loc)
        room.add_source(voice_loc, signal=voices_data[voice_idx][0])

        room.image_source_model(use_libroom=True)
        room.simulate()
        fg_signals = room.mic_array.signals[:, :total_samples]
        fg_target = np.clip(np.random.normal(FG_TARGET_VOL, FG_VOL_STD), 0, 1)
        fg_signals = fg_signals * fg_target / fg_signals.max()
        all_fg_signals.append(fg_signals)

    # BG
    if bg is not None:
        bg_radius = np.random.uniform(low=10.0, high=20.0)
        bg_theta = np.random.uniform(low=0, high=2 * np.pi)
        bg_loc = [bg_radius * np.cos(bg_theta), bg_radius * np.sin(bg_theta)]

        # Bg should be further away to be diffuse
        left_wall = np.random.uniform(low=-40, high=-20)
        right_wall = np.random.uniform(low=20, high=40)
        top_wall = np.random.uniform(low=20, high=40)
        bottom_wall = np.random.uniform(low=-40, high=-20)
        corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
                        [   right_wall, top_wall], [right_wall, bottom_wall]]).T
        absorption = np.random.uniform(low=0.5, high=0.99)
        room = pra.Room.from_corners(corners,
                                     fs=args.sr,
                                     max_order=10,
                                     absorption=absorption)
        mic_array = generate_mic_array(room, args.mic_radius, args.n_mics)
        room.add_source(bg_loc, signal=sample_bg)

        room.image_source_model(use_libroom=True)
        room.simulate()
        bg_signals = room.mic_array.signals[:, :total_samples]
        bg_target = np.clip(np.random.normal(BG_TARGET_VOL, BG_VOL_STD), 0, 1)
        bg_signals = bg_signals * bg_target / bg_signals.max()

    # Save
    for mic_idx in range(args.n_mics):
        output_prefix = str(
            Path(output_prefix_dir) / "mic{:02d}_".format(mic_idx))

        # Save FG
        all_fg_buffer = np.zeros((total_samples))
        for voice_idx in range(args.n_voices):
            curr_fg_buffer = np.pad(all_fg_signals[voice_idx][mic_idx],
                                    (0, total_samples))[:total_samples]
            sf.write(output_prefix + "voice{:02d}.wav".format(voice_idx),
                     curr_fg_buffer, args.sr)
            all_fg_buffer += curr_fg_buffer

        if bg is not None:
            bg_buffer = np.pad(bg_signals[mic_idx],
                               (0, total_samples))[:total_samples]
            sf.write(output_prefix + "bg.wav", bg_buffer, args.sr)

            sf.write(output_prefix + "mixed.wav", all_fg_buffer + bg_buffer,
                     args.sr)
        else:
            sf.write(output_prefix + "mixed.wav", all_fg_buffer,
                     args.sr)

    # [6]
    metadata = {}
    for voice_idx in range(args.n_voices):
        metadata['voice{:02d}'.format(voice_idx)] = {
            'position': voice_positions[voice_idx],
            'speaker_id': voices_data[voice_idx][1]
        }

    if bg is not None:
        metadata['bg'] = {'position': bg_loc}

    metadata_file = str(Path(output_prefix_dir) / "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


def main(args: argparse.Namespace):
    np.random.seed(args.seed)

    # Preload background to save time
    if args.input_background_path:
        background, _ = librosa.core.load(args.input_background_path,
                                          sr=args.sr,
                                          mono=True)
    else:
        background = None

    all_voices = Path(args.input_voice_dir).rglob('*.wav')
    args.all_voices = list(all_voices)

    pbar = tqdm.tqdm(total=args.n_outputs)
    pool = mp.Pool(args.n_workers)
    callback_fn = lambda _: pbar.update()
    for i in range(args.n_outputs):
        pool.apply_async(generate_sample,
                         args=(args, background, i),
                         callback=callback_fn,
                         error_callback=handle_error)
    pool.close()
    pool.join()
    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_voice_dir',
                        type=str,
                        help="Directory with voice wav files")
    parser.add_argument('output_path', type=str, help="Output directory to write the synthetic dataset")
    parser.add_argument('--input_background_path',
                        type=str)
    parser.add_argument('--n_mics', type=int, default=4)
    parser.add_argument('--mic_radius',
                        default=.03231,
                        type=float,
                        help="To do")
    parser.add_argument('--n_voices', type=int, default=4)
    parser.add_argument('--n_outputs', type=int, default=10000)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--duration', type=float, default=3.0)
    main(parser.parse_args())
