import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rescale_conv(conv, reference):
    """
    Rescale a convolutional module with `reference`.
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    """
    Rescale a module with `reference`.
    """
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def center_trim(tensor, reference):
    """
    Trim a tensor to match with the dimension of `reference`.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., diff // 2:-(diff - diff // 2)]
    return tensor


def left_trim(tensor, reference):
    """
    Trim a tensor to match with the dimension of `reference`. Trims only the end.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., 0:-diff]
    return tensor

def normalize_input(data):
    """
    Normalizes the input to have mean 0 std 1 for each input
    Inputs:
        data - torch.tensor of size batch x n_mics x n_samples
    """
    data = (data * 2**15).round() / 2**15
    ref = data.mean(1)  # Average across the n microphones
    means = ref.mean(1).unsqueeze(1).unsqueeze(2)
    stds = ref.std(1).unsqueeze(1).unsqueeze(2)
    data = (data - means) / stds

    return data, means, stds

def unnormalize_input(data, means, stds):
    """
    Unnormalizes the step done in the previous function
    """
    data = (data * stds.unsqueeze(3) + means.unsqueeze(3))
    return data


class CoSNetwork(nn.Module):
    """
    Cone of Silence network based on the Demucs network for audio source separation.
    """
    def __init__(
            self,
            n_audio_channels: int = 4,  # pylint: disable=redefined-outer-name
            window_conditioning_size: int = 5,
            kernel_size: int = 8,
            stride: int = 4,
            context: int = 3,
            depth: int = 6,
            channels: int = 64,
            growth: float = 2.0,
            lstm_layers: int = 2,
            rescale: float = 0.1):  # pylint: disable=redefined-outer-name
        super().__init__()
        self.n_audio_channels = n_audio_channels
        self.window_conditioning_size = window_conditioning_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.context = context
        self.depth = depth
        self.channels = channels
        self.growth = growth
        self.lstm_layers = lstm_layers
        self.rescale = rescale

        self.encoder = nn.ModuleList()  # Source encoder
        self.decoder = nn.ModuleList()  # Audio output decoder

        activation = nn.GLU(dim=1)

        in_channels = n_audio_channels  # Number of input channels

        # Wave U-Net structure
        for index in range(depth):
            encode = nn.ModuleDict()
            encode["conv1"] = nn.Conv1d(in_channels, channels, kernel_size,
                                        stride)
            encode["relu"] = nn.ReLU()

            encode["conv2"] = nn.Conv1d(channels, 2 * channels, 1)
            encode["activation"] = activation

            encode["gc_embed1"] = nn.Conv1d(self.window_conditioning_size, channels, 1)
            encode["gc_embed2"] = nn.Conv1d(self.window_conditioning_size, 2 * channels, 1)

            self.encoder.append(encode)

            decode = nn.ModuleDict()
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = 2 * n_audio_channels

            decode["conv1"] = nn.Conv1d(channels, 2 * channels, context)
            decode["activation"] = activation
            decode["conv2"] = nn.ConvTranspose1d(channels, out_channels,
                                                 kernel_size, stride)

            decode["gc_embed1"] = nn.Conv1d(self.window_conditioning_size, 2 * channels, 1)
            decode["gc_embed2"] = nn.Conv1d(self.window_conditioning_size, out_channels, 1)

            if index > 0:
                decode["relu"] = nn.ReLU()
            self.decoder.insert(0,
                                decode)  # Put it at the front, reverse order

            in_channels = channels
            channels = int(growth * channels)

        # Bi-directional LSTM for the bottleneck layer
        channels = in_channels
        self.lstm = nn.LSTM(bidirectional=True,
                            num_layers=lstm_layers,
                            hidden_size=channels,
                            input_size=channels)
        self.lstm_linear = nn.Linear(2 * channels, channels)

        rescale_module(self, reference=rescale)

    def forward(self, mix: torch.Tensor, angle_conditioning: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Forward pass. Note that in our current work the use of `locs` is disregarded.

        Args:
            mix (torch.Tensor) - An input recording of size `(batch_size, n_mics, time)`.

        Output:
            x - A source separation output at every microphone
        """
        x = mix
        saved = [x]

        # Encoder
        for encode in self.encoder:
            x = encode["conv1"](x)  # Conv 1d
            embedding = encode["gc_embed1"](angle_conditioning.unsqueeze(2))

            x = encode["relu"](x + embedding)
            x = encode["conv2"](x)

            embedding2 = encode["gc_embed2"](angle_conditioning.unsqueeze(2))
            x = encode["activation"](x + embedding2)
            saved.append(x)

        # Bi-directional LSTM at the bottleneck layer
        x = x.permute(2, 0, 1)  # prep input for LSTM
        self.lstm.flatten_parameters()  # to improve memory usage.
        x = self.lstm(x)[0]
        x = self.lstm_linear(x)
        x = x.permute(1, 2, 0)

        # Source decoder
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x)
            x = x + skip

            x = decode["conv1"](x)
            embedding = decode["gc_embed1"](angle_conditioning.unsqueeze(2))
            x = decode["activation"](x + embedding)
            x = decode["conv2"](x)
            embedding2 = decode["gc_embed2"](angle_conditioning.unsqueeze(2))
            if "relu" in decode:
                x = decode["relu"](x + embedding2)

        # Reformat the output
        x = x.view(x.size(0), 2, self.n_audio_channels, x.size(-1))

        return x

    def loss(self, voice_signals, gt_voice_signals):
        """Simple L1 loss between voice and gt"""
        return F.l1_loss(voice_signals, gt_voice_signals)

    def valid_length(self, length: int) -> int:  # pylint: disable=redefined-outer-name
        """
        Find the length of the input to the network such that the output's length is
        equal to the given `length`.
        """
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1

        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        return int(length)


def load_pretrain(model, state_dict):  # pylint: disable=redefined-outer-name
    """Loads the pretrained keys in state_dict into model"""
    for key in state_dict.keys():
        try:
            _ = model.load_state_dict({key: state_dict[key]}, strict=False)
            print("Loaded {} (shape = {}) from the pretrained model".format(
                key, state_dict[key].shape))
        except Exception as e:
            print("Failed to load {}".format(key))
            print(e)
