"""
The main separation by localization inference algorithm
"""

import argparse
import os

from collections import namedtuple

import librosa
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf

import cos.helpers.utils as utils

from cos.helpers.constants import ALL_WINDOW_SIZES, \
    FAR_FIELD_RADIUS
from cos.helpers.visualization import draw_diagram
from cos.training.network import CoSNetwork, center_trim, \
    normalize_input, unnormalize_input
from cos.helpers.eval_utils import cheap_sdr

# Constants which may be tweaked based on your setup
ENERGY_CUTOFF = 0.001
NMS_RADIUS = np.pi / 8
NMS_SIMILARITY_SDR = -5.0  # SDR cutoff for different candidates

CandidateVoice = namedtuple("CandidateVoice", ["angle", "energy", "data"])


def nms(candidate_voices, nms_cutoff):
    """
    Runs non-max suppression on the candidate voices
    """
    final_proposals = []
    initial_proposals = candidate_voices

    while len(initial_proposals) > 0:
        new_initial_proposals = []
        sorted_candidates = sorted(initial_proposals,
                                   key=lambda x: x[1],
                                   reverse=True)

        # Choose the loudest voice
        best_candidate_voice = sorted_candidates[0]
        final_proposals.append(best_candidate_voice)
        sorted_candidates.pop(0)

        # See if any of the rest should be removed
        for candidate_voice in sorted_candidates:
            different_locations = utils.angular_distance(
                candidate_voice.angle, best_candidate_voice.angle) > NMS_RADIUS

            different_content = abs(
                candidate_voice.data -
                best_candidate_voice.data).mean() > nms_cutoff

            different_content = cheap_sdr(
                candidate_voice.data[0],
                best_candidate_voice.data[0]) < nms_cutoff

            if different_locations:
                new_initial_proposals.append(candidate_voice)

        initial_proposals = new_initial_proposals

    return final_proposals


def forward_pass(model, target_angle, mixed_data, conditioning_label, args):
    """
    Runs the network on the mixed_data
    with the candidate region given by voice
    """
    target_pos = np.array([
        FAR_FIELD_RADIUS * np.cos(target_angle),
        FAR_FIELD_RADIUS * np.sin(target_angle)
    ])

    data, _ = utils.shift_mixture(
        torch.tensor(mixed_data).to(args.device), target_pos, args.mic_radius,
        args.sr)
    data = data.float().unsqueeze(0)  # Batch size is 1

    # Normalize input
    data, means, stds = normalize_input(data)

    # Run through the model
    valid_length = model.valid_length(data.shape[-1])
    delta = valid_length - data.shape[-1]
    padded = F.pad(data, (delta // 2, delta - delta // 2))

    output_signal = model(padded, conditioning_label)
    output_signal = center_trim(output_signal, data)

    output_signal = unnormalize_input(output_signal, means, stds)
    output_voices = output_signal[:, 0]  # batch x n_mics x n_samples

    output_np = output_voices.detach().cpu().numpy()[0]
    energy = librosa.feature.rms(output_np).mean()

    return output_np, energy


def run_separation(mixed_data, model, args,
                   energy_cutoff=ENERGY_CUTOFF,
                   nms_cutoff=NMS_SIMILARITY_SDR): # yapf: disable
    """
    The main separation by localization algorithm
    """
    # Get the initial candidates
    num_windows = len(ALL_WINDOW_SIZES)
    starting_angles = utils.get_starting_angles(ALL_WINDOW_SIZES[0])
    candidate_voices = [CandidateVoice(x, None, None) for x in starting_angles]

    # All steps of the binary search
    for window_idx in range(num_windows):
        if args.debug:
            print("---------")
        conditioning_label = torch.tensor(utils.to_categorical(
            window_idx, 5)).float().to(args.device).unsqueeze(0)

        curr_window_size = ALL_WINDOW_SIZES[window_idx]
        new_candidate_voices = []

        # Iterate over all the potential locations
        for voice in candidate_voices:
            output, energy = forward_pass(model, voice.angle, mixed_data,
                                          conditioning_label, args)

            if args.debug:
                print("Angle {:.2f} energy {}".format(voice.angle, energy))
                fname = "out{}_angle{:.2f}.wav".format(
                    window_idx, voice.angle * 180 / np.pi)
                sf.write(os.path.join(args.writing_dir, fname), output[0],
                         args.sr)

            # If there was something there
            if energy > energy_cutoff:

                # We're done searching so undo the shifts
                if window_idx == num_windows - 1:
                    target_pos = np.array([
                        FAR_FIELD_RADIUS * np.cos(voice.angle),
                        FAR_FIELD_RADIUS * np.sin(voice.angle)
                    ])
                    unshifted_output, _ = utils.shift_mixture(output,
                                                           target_pos,
                                                           args.mic_radius,
                                                           args.sr,
                                                           inverse=True)

                    new_candidate_voices.append(
                        CandidateVoice(voice.angle, energy, unshifted_output))

                # Split region and recurse.
                # This division has more redundancy than necessary, but 
                # Avoids missing some sources
                else:
                    new_candidate_voices.append(
                        CandidateVoice(
                            voice.angle + curr_window_size / 3,
                            energy, output))
                    new_candidate_voices.append(
                        CandidateVoice(
                            voice.angle - curr_window_size / 3,
                            energy, output))
                    new_candidate_voices.append(
                        CandidateVoice(
                            voice.angle,
                            energy, output))

        candidate_voices = new_candidate_voices

    # Run NMS on the final output and return
    return nms(candidate_voices, nms_cutoff)


def main(args):
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')

    args.device = device
    model = CoSNetwork(n_audio_channels=args.n_channels)
    model.load_state_dict(torch.load(args.model_checkpoint), strict=True)
    model.train = False
    model.to(device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mixed_data = librosa.core.load(args.input_file, mono=False, sr=args.sr)[0]
    assert mixed_data.shape[0] == args.n_channels

    temporal_chunk_size = int(args.sr * args.duration)
    num_chunks = (mixed_data.shape[1] // temporal_chunk_size) + 1

    for chunk_idx in range(num_chunks):
        curr_writing_dir = os.path.join(args.output_dir,
                                        "{:03d}".format(chunk_idx))
        if not os.path.exists(curr_writing_dir):
            os.makedirs(curr_writing_dir)

        args.writing_dir = curr_writing_dir
        curr_mixed_data = mixed_data[:, (chunk_idx *
                                         temporal_chunk_size):(chunk_idx + 1) *
                                     temporal_chunk_size]

        output_voices = run_separation(curr_mixed_data, model, args)
        for voice in output_voices[:1]:
            fname = "output_angle{:.2f}.wav".format(
                voice.angle * 180 / np.pi)
            sf.write(os.path.join(args.writing_dir, fname), voice.data[0],
                     args.sr)

        candidate_angles = [voice.angle for voice in output_voices]
        draw_diagram([], candidate_angles,
                    ALL_WINDOW_SIZES[-1],
                    os.path.join(args.writing_dir, "positions.png".format(chunk_idx)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_checkpoint',
                        type=str,
                        help="Path to the model file")
    parser.add_argument('input_file', type=str, help="Path to the input file")
    parser.add_argument('output_dir',
                        type=str,
                        help="Path to write the outputs")
    parser.add_argument('--sr', type=int, default=22050, help="Sampling rate")
    parser.add_argument('--n_channels',
                        type=int,
                        default=2,
                        help="Number of channels")
    parser.add_argument('--use_cuda',
                        dest='use_cuda',
                        action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--debug',
                        action='store_true',
                        help="Save intermediate outputs")
    parser.add_argument('--mic_radius',
                        default=.03231,
                        type=float,
                        help="Radius of the mic array")
    parser.add_argument('--duration',
                        default=3.0,
                        type=float,
                        help="Seconds of input to the network")
    main(parser.parse_args())
