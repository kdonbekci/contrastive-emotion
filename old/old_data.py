import webdataset as wds
import numpy as np
import io
import soundfile as sf
import json
import torch
from collections import namedtuple
import random
from torch_pitch_shift import get_fast_shifts, pitch_shift
from braceexpand import braceexpand
import os
import time


class SampledShards(torch.utils.data.IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls: list | str,
        nshards: int,
        seed=0,
        worker_seed=None,
        deterministic=False,
    ):
        super().__init__()
        if isinstance(urls, str):
            urls = braceexpand(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.worker_seed = (
            wds.utils.pytorch_worker_seed if worker_seed is None else worker_seed
        )
        self.deterministic = deterministic
        self.seed = seed
        self.epoch = -1

    def __iter__(self):
        """Return an iterator over the shards."""
        self.epoch += 1
        if self.deterministic:
            seed = wds.utils.make_seed(self.worker_seed(), self.epoch, self.seed)
        else:
            seed = wds.utils.make_seed(
                self.worker_seed(),
                self.epoch,
                self.seed,
                os.getpid(),
                time.time_ns(),
                os.urandom(4),
            )
        if os.environ.get("WDS_SHOW_SEED", "0") == "1":
            print(f"# SampledShards seed {seed}")
        np.random.seed(seed)
        indexes = np.random.choice(
            np.arange(len(self.urls)), size=self.nshards, replace=False
        )
        for ix in indexes:
            yield dict(url=self.urls[ix])


SAMPLE_RATE = 16_000
MSP_PODCAST_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
    "contempt",
    "other",
    "no_agreement",
]
MSP_PODCAST_EMOTION_TO_IX = {
    emotion: i for i, emotion in enumerate(MSP_PODCAST_EMOTIONS)
}
MSP_PODCAST_IX_TO_EMOTION = {
    i: emotion for emotion, i in MSP_PODCAST_EMOTION_TO_IX.items()
}

IEMOCAP_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
    "excited",
    "frustrated",
    "other",
    "unknown",
]
IEMOCAP_EMOTION_TO_IX = {emotion: i for i, emotion in enumerate(IEMOCAP_EMOTIONS)}
IEMOCAP_IX_TO_EMOTION = {i: emotion for emotion, i in IEMOCAP_EMOTION_TO_IX.items()}

RAVDESS_EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]
RAVDESS_EMOTION_TO_IX = {emotion: i for i, emotion in enumerate(RAVDESS_EMOTIONS)}
RAVDESS_IX_TO_EMOTION = {i: emotion for emotion, i in RAVDESS_EMOTION_TO_IX.items()}
RAVDESS_TRANSCRIPTS = ["Kids are talking by the door", "Dogs are sitting by the door"]


def decode(dataset: str, ix_to_label, decode_json: bool):
    if dataset == "MSP_PODCAST":

        def fn(datum):
            wav, sr = sf.read(io.BytesIO(datum["flac"]), dtype="float32")
            del datum["flac"]
            assert SAMPLE_RATE == sr
            datum["wav"] = wav
            if decode_json:
                json_data = json.loads(datum["json"])
            else:
                json_data = datum["json"]
            del datum["json"]
            for key, val in json_data.items():
                datum[key] = val
            datum["emotion_ix"] = datum["emotion"]
            datum["emotion"] = ix_to_label[datum["emotion"]]
            datum["key"] = datum["__key__"]
            datum["url"] = datum["__url__"]
            return datum

    elif dataset == "IEMOCAP_audio":

        def fn(datum):
            wav, sr = sf.read(io.BytesIO(datum["flac"]), dtype="float32")
            del datum["flac"]
            assert SAMPLE_RATE == sr
            datum["wav"] = wav
            if decode_json:
                json_data = json.loads(datum["json"])
            else:
                json_data = datum["json"]
            del datum["json"]
            for key, val in json_data.items():
                datum[key] = val
            datum["emotion_ix"] = datum["emotion"]
            datum["gender"] = datum["speaker"].split("_")[1]
            datum["emotion"] = ix_to_label[datum["emotion"]]
            datum["key"] = datum["__key__"]
            datum["url"] = datum["__url__"]
            return datum

    elif dataset == "RAVDESS_audio":

        def fn(datum):
            wav, sr = sf.read(io.BytesIO(datum["flac"]), dtype="float32")
            if len(wav.shape) == 2:
                wav = wav[:, 0]
            del datum["flac"]
            assert SAMPLE_RATE == sr
            datum["wav"] = wav
            if decode_json:
                json_data = json.loads(datum["json"])
            else:
                json_data = datum["json"]
            del datum["json"]
            for key, val in json_data.items():
                datum[key] = val
            datum["emotion_ix"] = datum["emotion"]
            # odd numbers are male
            datum["gender"] = "M" if datum["speaker"] % 2 == 1 else "F"
            datum["emotion"] = ix_to_label[datum["emotion"]]
            datum["key"] = datum["__key__"]
            datum["url"] = datum["__url__"]
            datum["transcript"] = RAVDESS_TRANSCRIPTS[datum["transcript"]]
            return datum

    return fn


def apply_augmentation(augmentation, p=1.0, in_key="wav", out_key="wav_aug", **kwargs):
    def fn(datum):
        if random.random() < p:
            datum[out_key] = augmentation(datum[in_key], **kwargs)
        else:
            datum[out_key] = datum[in_key]
        return datum

    return fn


def crop(crop_duration: float, random: bool, keys: list):
    max_samples = int(crop_duration * SAMPLE_RATE)

    def fn(datum):
        for key in keys:
            num_samples = datum[key].shape[0]
            if num_samples >= max_samples:
                if random:
                    start = np.random.randint(0, num_samples - max_samples + 1)
                else:
                    start = 0
                end = start + max_samples
            else:
                start = 0
                end = num_samples
            datum[key] = datum[key][start:end]
        return datum

    return fn


def collate_and_featurize(
    list_keys: list,
    tensor_keys: list,
    wav_keys: list,
    feature_keys: list,
    feature_extractor,
    feature_kwargs: dict,
):
    if list_keys is None:
        list_keys = []
    if tensor_keys is None:
        tensor_keys = []
    if wav_keys is None:
        wav_keys = []
    if feature_keys is None:
        feature_keys = []
    if feature_extractor is not None:
        feature_extractor = feature_extractor(**feature_kwargs)

    feature_wav_keys = set(feature_keys + wav_keys)
    num_samples_keys = {
        (
            f"num_samples"
            if key == "wav"
            else f"num_samples_{key.removeprefix('wav_')}"
        ): key
        for key in wav_keys
    }
    attention_mask_keys = {
        (
            "attention_mask"
            if key == "wav"
            else f"attention_mask_{key.removeprefix('wav_')}"
        ): key
        for key in feature_keys
    }
    feature_keys = {
        ("feats" if key == "wav" else f"feats_{key.removeprefix('wav_')}"): key
        for key in feature_keys
    }

    Batch = namedtuple(
        "Batch",
        list_keys
        + tensor_keys
        + wav_keys
        + list(num_samples_keys.keys())
        + list(feature_keys.keys())
        + list(attention_mask_keys.keys()),
    )

    def fn(batch):
        lists = {key: [] for key in list_keys}
        tensors = {key: [] for key in tensor_keys}

        wavs = {key: [] for key in feature_wav_keys}
        max_samples = {key: -1 for key in wav_keys}
        num_samples = {key: [] for key in wav_keys}
        for datum in batch:
            for key in list_keys:
                lists[key].append(datum[key])
            for key in tensor_keys:
                tensors[key].append(datum[key])
            for key in feature_wav_keys:
                wavs[key].append(datum[key])
            for key in wav_keys:
                max_samples[key] = max(max_samples[key], datum[key].shape[0])
                num_samples[key].append(datum[key].shape[0])

        tensors = {key: torch.tensor(val) for key, val in tensors.items()}
        feats = {
            wav_key: feature_extractor(
                wavs[wav_key],
                sampling_rate=SAMPLE_RATE,
                return_attention_mask=True,
                return_tensors="pt",
            )
            for wav_key in feature_keys.values()
        }
        attention_mask = {
            attention_mask_key: feats[wav_key].attention_mask
            for attention_mask_key, wav_key in attention_mask_keys.items()
        }
        feats = {
            feat_key: feats[wav_key].input_features
            for feat_key, wav_key in feature_keys.items()
        }

        wavs = {
            key: torch.tensor(
                np.array(
                    [np.pad(x, (0, max_samples[key] - x.shape[0])) for x in wavs[key]]
                )
            )
            for key in wav_keys
        }
        num_samples = {
            num_samples_key: torch.tensor(num_samples[wav_key])
            for num_samples_key, wav_key in num_samples_keys.items()
        }
        batched = lists
        batched.update(tensors)
        batched.update(wavs)
        batched.update(num_samples)
        batched.update(feats)
        batched.update(attention_mask)
        return batched

    return fn


class Cache:
    def __init__(self, shuffle: bool):
        self.storage = []
        self.storage_full = False
        self.shuffle = shuffle

    def __call__(self, src):
        if not self.storage_full:
            self.storage = []
            for datum in src:
                self.storage.append(datum)
                yield datum
            self.storage_full = True
        else:
            for datum in self.storage:
                yield datum


class FilterLabels:
    def __init__(self, labels_wanted, labels_to_ix, label_key="emotion"):
        assert set(labels_wanted).issubset(set(labels_to_ix.keys()))
        self.labels = labels_wanted
        self.label_to_ix = {label: ix for ix, label in enumerate(self.labels)}
        self.ix_to_label = {ix: label for label, ix in self.label_to_ix.items()}
        self._ix_remap = {
            labels_to_ix[label]: new_ix for new_ix, label in enumerate(self.labels)
        }
        self._label_key = label_key
        self._label_ix_wanted = set([labels_to_ix[label] for label in self.labels])

    def __call__(self, src):
        for datum in src:
            datum["json"] = json.loads(datum["json"])
            if (old_ix := datum["json"][self._label_key]) in self._label_ix_wanted:
                datum["json"][self._label_key] = self._ix_remap[old_ix]
                yield datum


class PitchShift:
    def __init__(self, device: str, p: float = 1.0, sample_rate: int = SAMPLE_RATE):
        self.ratios = []
        self.sample_rate = sample_rate
        self.p = p
        self.device = device
        for ratio in get_fast_shifts(sample_rate):
            x = ratio.__float__()
            if x < 0.7 or x > 1.3:
                continue
            self.ratios.append(ratio)

    def __call__(self, wav):
        # wav is unbatched numpy array
        if random.random() > self.p:
            return wav

        ratio = random.choice(self.ratios)
        out = (
            pitch_shift(
                torch.as_tensor(wav[None, None, :]).to(self.device),
                ratio,
                sample_rate=self.sample_rate,
            )
            .cpu()[0, 0, :]
            .numpy()
        )
        return out


class Compose:
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, src, *args, **kwargs):
        for aug in self.augs:
            src = aug(src, *args, **kwargs)
        return src
