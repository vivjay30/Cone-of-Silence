# The Cone of Silence: Speech Separation by Localization
![alt text](https://drive.google.com/uc?export=view&id=1h5ffwTdyD8M9ujrDIlbUl5s9SG7Ht1pd)

[Teerapat Jenrungrot](https://mjenrungrot.com/)**\***, [Vivek Jayaram](http://www.vivekjayaram.com/)**\***, [Steve Seitz](https://homes.cs.washington.edu/~seitz/), and [Ira Kemelmacher-Shlizerman](https://homes.cs.washington.edu/~kemelmi/)
*\*Co-First Authors*
University of Washington

### [Project Page](http://grail.cs.washington.edu/projects/cone-of-silence/)
Video and audio demos are available at the project page

### Paper (Arxiv) - Coming Soon
### Blog Post - Coming Soon

### Summary
Our method performs source separation and localization for human speakers. Key features include handling an arbitary number of speakers and moving speakers with a single network. This code allows you to run and evaluate our method on synthetically rendered data. If you have a multi-microphone array, you can also obtain real results like the ones in our demo video.

## Getting Started
Clone the repository:
```
git clone https://github.com/vivjay30/Cone-of-Silence.git
cd Cone-of-Silence
export PYTHONPATH=$PYTHONPATH:`pwd`
```

Make sure all the requirements in the requirements.txt are installed. We tested the code with torch 1.3.0, librosa 0.7.0 and cuda 10.0

Download Pretrained Models: (Todo)

## Quickstart: Running on Real Data
You can easily produce results like those in our demo videos. Our pre-trained real models work with the 4 mic [Seed ReSpeaker MicArray v 2.0](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/). When you capture the data, it must be a m channel recording. Run the full command like below. For moving sources, reduce the duration flag to 1.5 and add `--moving` to stop the search at a coarse window.
```
python cos/inference/separation_by_localization.py /path/to/model.pt /path/to/input_file.wav outputs/some_dirname/ --n_channels 4 --sr 44100 --mic_radius .03231 --use_cuda
```



## Rendering Synthetic Spatial Data
For training and evaluation, we use synthetically rendered spatial data. We place the voices in a virtual room and render the arrival times, level differences, and reverb using pyroomacoustics. We used the VCTK dataset but any voice dataset would work. An example command is below
```
python cos/generate_dataset.py /path/to/VCTK/data ./output/somename --input_background_path any_bg_audio.wav --n_voices 2  --n_outputs 1000 --mic_radius {radius} --n_mics {M}
```

## Training on Synthetic Data
Below is an example command to train on the rendered data. You need to replace the training and testing dirs with the path to the generated datasets from above. We highly recommend initializing with a pre-trained model and not training from scratch.
```
python cos/training/train.py ./generated/train_dir ./generated/test_dir --name experiment_name --checkpoints_dir ./checkpoints --pretrain_path ./path/to/pretrained.pt --batch_size 8 --mic_radius {radius}  --n_mics {M} --use_cuda
```

## Training on Real Data
For those looking to improve on the pretrained models, we recommend gathering a lot more real data. We did not have the ability to gather very accurately positioned real data in a proper sound chamber. By training with a lot more real data, the results will almost certainly improve. All you have to do is create synthetic composites of speakers in the same format as the synthetic data, and run the same training script.

## Evaluation
For the synthetic data and evaluation, we use a setup of 6 mics in a circle of radius 7.25cm. The following is instructions to obtain results on mixtures of N voices and no backgrounds. First generate a synthetic datset with the microphone setup specified previous with ```--n_voices 8``` from the test set of VCTK. Then run the following script:  

```python cos/inference/evaluate_synthetic.py /path/to/rendered_data/ /path/to/model.pt --n_channels 6 --mic_radius .0725 --sr 44100 --use_cuda --n_workers 1 --n_voices {N}```

Add ```--compute_sdr``` separately to get the SDR.

| Number of Speakers N | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|----------------------|-------|-------|-------|-------|-------|-------|-------|
| Median SI-SDRi (dB)  | 13.9  | 13.2  | 12.2  | 10.8  | 9.1   | 7.2   | 6.3   |
| Median Angular Error | 2.0   | 2.3   | 2.7   | 3.5   | 4.4   | 5.2   | 6.3   |
| Precision            | 0.947 | 0.936 | 0.897 | 0.912 | 0.932 | 0.936 | 0.966 |
| Recall               | 0.979 | 0.972 | 0.915 | 0.898 | 0.859 | 0825  | 0.785 |


