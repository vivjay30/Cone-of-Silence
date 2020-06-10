# The Cone of Silence: Separation by Localization

Our method performs source separation and localization for human speakers as described in our paper. This code allows you to run and evaluate our method on synthetically rendered data. If you have a multi-microphone array, you can also obtain real results like the ones in our demo video. Our pre-trained models work with the 4 mic [Seed ReSpeaker MicArray v 2.0](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) or a synthetic 6 microphone array of radius 7.25cm.

### Requirements
librosa
torch
soundfile
scipy
pyroomacoustics
matplotlib

cd into the directory and run
```export PYTHONPATH=$PYTHONPATH:`pwd` ```

### Rendering the Data

The default mic setup is for the ReSpeaker 4 mic array mentioned above. For numerical experiments, we used a slighly different synthetic setup as mentioned previously.
```python cos/generate_dataset.py /path/to/VCTK/data ./output/somename --input_background_path any_bg_sounds.wav --n_voices 2  --n_outputs 1000 --mic_radius {radius} --n_mics {M}```

### Training on Synthetic Data
Below is an example command to train on the rendered data. You need to replace the training and testing dirs with the path to the generated datasets from above.
```python cos/training/train.py ./generated/train_dir ./generated/test_dir --name experiment_name --checkpoints_dir ./checkpoints --pretrain_path ./path/to/pretrained.pt --batch_size 8 --mic_radius .03231  --n_mics 4 --use_cuda```

### Running on Real Data
When you capture the data, it must be a m channel recording from a microphone with the same configuration as used during training. If you want to support moving sources, change the DURATION variable to run on smaller time inputs.
```python cos/inference/separation_by_localization.py /path/to/model.pt /path/to/input_file.wav outputs/some_dirname/ --n_channels 4 --sr 44100 --mic_radius .03231 --use_cuda```

### Evaluation
For the synthetic data and evaluation, we use a setup of 6 mics in a circle of radius 7.25cm. The following is instructions to obtain results on mixtures of N voices and no backgrounds. First generate a synthetic datset with the microphone setup specified previous with ```--n_voices 8``` from the test set of VCTK. Then run the following script:  

```python cos/inference/evaluate_synthetic.py /path/to/rendered_data/ /path/to/model.pt --n_channels 6 --mic_radius .0725 --sr 22050 --use_cuda --n_workers 1 --n_voices {N}```

Add ```--compute_sdr``` separately to get the SDR.

| Number of Speakers N | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|----------------------|-------|-------|-------|-------|-------|-------|-------|
| Median SI-SDRi (dB)  | 13.9  | 13.2  | 12.2  | 10.8  | 9.1   | 7.2   | 6.3   |
| Median Angular Error | 2.0   | 2.3   | 2.7   | 3.5   | 4.4   | 5.2   | 6.3   |
| Precision            | 0.947 | 0.936 | 0.897 | 0.912 | 0.932 | 0.936 | 0.966 |
| Recall               | 0.979 | 0.972 | 0.915 | 0.898 | 0.859 | 0825  | 0.785 |


