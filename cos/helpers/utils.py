"""A collection of useful helper functions"""

import numpy as np
import torch

from cos.helpers.constants import SPEED_OF_SOUND


def shift_mixture(input_data, target_position, mic_radius, sr, inverse=False):
    """
    Shifts the input according to the voice position. This
    lines up the voice samples in the time domain coming from a target_angle
    Args:
        input_data - M x T numpy array or torch tensor
        target_position - The location where the data should be aligned
        mic_radius - In meters. The number of mics is inferred from
            the input_Data
        sr - Sample Rate in samples/sec
        inverse - Whether to align or undo a previous alignment

    Returns: shifted data and a list of the shifts
    """
    num_channels = input_data.shape[0]

    # Must match exactly the generated or captured data
    mic_array = [[
        mic_radius * np.cos(2 * np.pi / num_channels * i),
        mic_radius * np.sin(2 * np.pi / num_channels * i)
    ] for i in range(num_channels)]

    # Mic 0 is the canonical position
    distance_mic0 = np.linalg.norm(mic_array[0] - target_position)
    shifts = [0]

    # Check if numpy or torch
    if isinstance(input_data, np.ndarray):
        shift_fn = np.roll
    elif isinstance(input_data, torch.Tensor):
        shift_fn = torch.roll
    else:
        raise TypeError("Unknown input data type: {}".format(type(input_data)))

    # Shift each channel of the mixture to align with mic0
    for channel_idx in range(1, num_channels):
        distance = np.linalg.norm(mic_array[channel_idx] - target_position)
        distance_diff = distance - distance_mic0
        shift_time = distance_diff / SPEED_OF_SOUND
        shift_samples = int(round(sr * shift_time))
        if inverse:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],
                                               shift_samples)
        else:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],
                                               -shift_samples)
        shifts.append(shift_samples)

    return input_data, shifts


def angular_distance(angle1, angle2):
    """
    Computes the distance in radians betwen angle1 and angle2.
    We assume they are between -pi and pi
    """
    d1 = abs(angle1 - angle2)
    d2 = abs(angle1 - angle2 + 2 * np.pi)
    d3 = abs(angle2 - angle1 + 2 * np.pi)

    return min(d1, d2, d3)

def get_starting_angles(window_size):
    """Returns the list of target angles for a window size"""
    divisor = int(round(2 * np.pi / window_size))
    return np.array(list(range(-divisor + 1, divisor, 2))) * np.pi / divisor


def to_categorical(index: int, num_classes: int):
    """Creates a 1-hot encoded np array"""
    data = np.zeros((num_classes))
    data[index] = 1
    return data


