import numpy as np

from pysndfx import AudioEffectsChain

class RandomAudioPerturbation(object):
    """Randomly perturb audio samples"""

    def __call__(self, data):
        """
        Data must be mics x T numpy array
        """
        highshelf_gain = np.random.normal(0, 2)
        lowshelf_gain = np.random.normal(0, 2)
        noise_amount = np.random.uniform(0, 0.001)

        fx = (
            AudioEffectsChain()
            .highshelf(gain=highshelf_gain)
            .lowshelf(gain=lowshelf_gain)
        )

        for i in range(data.shape[0]):
            data[i] = fx(data[i])
            data[i] += np.random.uniform(-noise_amount, noise_amount, size=data[i].shape)
        return data
