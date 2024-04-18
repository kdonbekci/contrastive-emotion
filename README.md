# Contrastive Learning for Speech Emotion Recognition

The repo contains two Jupyter notebooks. `contrastive.ipynb` has the pipeline to run self-supervised contrastive loss implemented and `supervised.ipynb` has the pipeline for supervised cross-entropy loss. 

In addition, `data.py` hosts the data-loading code which makes use of the DALI library by NVIDIA. This enables on the fly augmentations and feature processing to run on GPUs and is necessary to not bottleneck training. `old_data.py` contains code for data loading using the regular PyTorch DataLoader and Python multiprocessing, however with the inclusion of more and more augmentations, this became a bottleneck. The notebooks with prefix `old_` have the pipelines that were built using the regular PyTorch DataLoaders. `model.py` contains the model definition that supports loading different Whisper models, freezing/unfreezing parameters, and adjusting the context size of the model by trimming the positional encoding layer. By default, Whisper expects 30-second audio clips, with mel-filterbanks extracted over 3000 frames. By this modification, one can trim the model inputs to only 300 frames for 3 seconds of audio and reduce the VRAM requirements for the model inputs and activations drastically. 

The datasets are hosted in HuggingFace and the labels for emotions have been normalized in the following way:
1. First 7 emotions are the ones below for each dataset (emotion label 0-6) 
```
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted"]
```
2. For any additional label in the dataset, they are first sorted and populate index 7 and onwards. For instance MSP PODCAST has `contempt, no_agreement, other` emotion categories in addition and the full emotion to index mapping looks like:
```
{'neutral': 0,
 'happy': 1,
 'sad': 2,
 'angry': 3,
 'fearful': 4,
 'surprised': 5,
 'disgusted': 6,
 'contempt': 7,
 'no_agreement': 8,
 'other': 9}
```
3. Valence, arousal and dominance scores are normalized to be in the range of 0.0-1.0 using the original datasets scale. For instance MSP PODCAST used a 1-7 Likert scale and the scores were normalized via:
```
def map_scale(score: float):
    assert 1.0 <= score <= 7.0
    return (score - 1.0) / 6.0
```


## Next steps:

- [x] Transition to NVIDIA DALI
- [x] Map MSP PODCAST, IEMOCAP, RAVDESS to shared labels/scores
- [ ] Reduce target emotion set to match literature (4-class commonly reported) [DeepSpeech example](https://github.com/speechbrain/speechbrain/tree/develop/recipes/IEMOCAP)
- [ ] Negative pair mining using the valence, arousal, dominance

