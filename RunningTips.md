# Tips for Running on Real Data

## Hardware
We have only tested this network with circular microphone arrays. It might be possible to train a new model for ad-hoc or linear mic arrays. The pretrained model works with the 4 mic [Seed ReSpeaker MicArray v 2.0](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/)

## Capturing Data
### Positioning
For best results, the sources should not be too far from the microphone. Our model was trained with sources between 1-4 meters from the microphone. Extreme far field (>4m) has not been explored. Ideally the sources should be at roughly the same elevation angle as the microphone. For example, If the microphone is on the ground and the sources are standing, the assumptions in the pre-shift formulas no longer hold.
### Obstructions
T he point source model breaks down if there is not a direct line of sight between the source and the microphone, for example if there is a laptop between the voice and mic. 
## Hyperparameters
If the sources are completely stationary, you can increase the `--duration` flag which processes large chunks at a time. This improves the performance and reduces boundary effects. At the top of `cos/inference/separation_by_localization.py` are additional parameters. Tweak `ENERGY_CUTOFF` to more aggressively keep or reject sources. For more aggressive non-max suppression, you can reduce `NMS_SIMILARITY_SDR` which only keeps additional sources if they have a SDR to the existing source that is lower than this parameter. 

## Post Processing
After separating voices from each other, any type of single channel post processing can be run on the output. We found that it was useful to run a low-pass filter on the output. The network requires 44.1kHz sampling runing for localization and time differences, but can sometimes produce artifacts in these high frequency ranges. Because voice doesn't contain many frequences about 10kHz, they can simply be cut.

# Tips for Training on Real Data

We strongly recommend training on data collected from your specific microphone. We expect the network performance to improve with more training data, even beyond the pretrained models we have provided. If you train on new data, do not train from scratch: instead fine-tune from our existing weights even if the number of channels is different. Training from scratch often results in the network outputting silence everywhere. In our experience, it is best to jointly train on real and synthetic data. 
