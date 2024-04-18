from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as dali_math
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from pathlib import Path

SAMPLE_RATE = 16_000
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted"]


def get_max_frames(max_duration: float, hop_length: int):
    max_frames = max_duration * SAMPLE_RATE / hop_length
    assert max_frames % 1 == 0.0
    max_frames = int(max_frames)
    return max_frames


def get_urls_indexes(dataset_dir: str, random_shuffle: bool = True):
    dataset_dir = Path(dataset_dir).resolve()
    urls = {}
    indexes = {}
    is_shallow = None
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir():
            is_shallow = False
            split = child.stem
            urls[split] = []
            indexes[split] = []
            for tarfile in sorted(child.glob("*.tar")):
                urls[split].append(str(tarfile))
                indexes[split].append(str(tarfile.with_suffix(".txt")))
        else:
            assert child.suffix == ".txt" or child.suffix == ".tar"
            assert is_shallow is None
            is_shallow = True
            break
    if is_shallow:
        for tarfile in sorted(dataset_dir.glob("*.tar")):
            split = tarfile.stem
            urls[split] = [str(tarfile)]
            indexes[split] = [str(tarfile.with_suffix(".txt"))]
    if random_shuffle:
        for split in urls:
            urls[split], indexes[split] = shuffle(urls[split], indexes[split])
    return urls, indexes


@pipeline_def
def pipeline(
    urls: list[str],
    index_paths: list[str],
    num_shards: int = 1,
    shard_id: int = 0,
    cuda: bool = True,
    random_shuffle: bool = True,
    read_ahead: bool = False,
    peak_normalize: bool = True,
    preemphasis: bool = False,
    max_duration: float = 5.0,
    n_fft: int = 400,
    hop_length: int = 160,
    num_mel: int = 80,
    augmentations: list[str] = [],
    ref: bool = False,
    ref_augmentations: list[str] = [],
):
    pipe = Pipeline.current()
    max_frames = get_max_frames(max_duration=max_duration, hop_length=hop_length)
    audio_raw, emo, emodim = fn.readers.webdataset(
        ext=["flac", "emo", "emodim"],
        paths=urls,
        index_paths=index_paths,
        dtypes=[types.UINT8, types.UINT8, types.FLOAT],
        missing_component_behavior="error",
        pad_last_batch=True,
        random_shuffle=random_shuffle,
        read_ahead=read_ahead,
        name="webdataset",
        num_shards=num_shards,
        shard_id=shard_id,
    )
    emo = emo[0]
    wav, sr = fn.decoders.audio(audio_raw, device="cpu", downmix=True)
    num_samples = fn.shapes(wav)[0]
    if cuda:
        wav = wav.gpu()
    if peak_normalize:
        wav = wav / fn.reductions.max(dali_math.abs(wav))
    if preemphasis:
        wav = fn.preemphasis_filter(wav)

    num_frames_cpu = dali_math.floor(
        num_samples / hop_length + 1
    )  # in case need to use this on cpu
    spec = fn.spectrogram(
        wav,
        nfft=n_fft,
        window_length=n_fft,
        window_step=hop_length,
    )
    num_frames = fn.shapes(spec)[1]
    num_frames = dali_math.min(num_frames, max_frames)
    mel = fn.mel_filter_bank(
        spec,
        freq_high=8000,
        freq_low=0,
        nfilter=num_mel,
        normalize=True,
        sample_rate=SAMPLE_RATE,
    )
    # MATCHING WHISPER MEL PROCESSING
    mel = dali_math.clamp(mel, lo=1e-10, hi=np.finfo("float32").max)
    mel = dali_math.log10(mel)
    mel = dali_math.max(mel, fn.reductions.max(mel) - 8.0)
    mel = (mel + 4.0) / 4.0

    # PAD TO MAX_FRAME
    fill = fn.random.uniform(range=(-2.5, -1.5), shape=(1,))
    mel = fn.reinterpret(mel, layout="ft")
    mel = fn.pad(
        mel, axis_names="t", fill_value=fn.squeeze(fill, axes=0), shape=(max_frames,)
    )
    # RANDOM CROP
    mel = fn.reinterpret(fn.expand_dims(mel, axes=2), layout="HWC")
    if ref:
        mel_ref = fn.crop(
            mel,
            crop=(num_mel, max_frames),
            crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
            out_of_bounds_policy="error",
        )
        mel_ref = fn.reinterpret(fn.squeeze(mel_ref, axes=2), layout="ft")
    mel = fn.crop(
        mel,
        crop=(num_mel, max_frames),
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        out_of_bounds_policy="error",
    )
    mel = fn.reinterpret(fn.squeeze(mel, axes=2), layout="ft")

    for aug in augmentations:
        if aug == "spec_aug":
            # Frequency Masking
            mel = fn.erase(
                mel,
                anchor=fn.random.uniform(range=(0, num_mel)),
                shape=fn.random.uniform(range=(0, num_mel // 4)),
                normalized=False,
                axis_names="f",
                fill_value=fill,
            )
            # Time Masking
            mel = fn.erase(
                mel,
                anchor=fn.random.uniform(range=(0.0, 1.0)),
                shape=fn.random.uniform(
                    range=(0, int(1 * SAMPLE_RATE / hop_length))
                ),  # up to 1 sec
                normalized_anchor=True,
                normalized_shape=False,
                axis_names="t",
                fill_value=fill,
            )
        elif aug == "salt_pepper":
            mel = fn.noise.salt_and_pepper(mel, prob=0.05)
        elif aug == "white_noise":
            mel = fn.noise.gaussian(mel, mean=0, stddev=0.1)
        else:
            raise ValueError(f"{aug=} not recognized")
    if ref and ref_augmentations:
        for aug in ref_augmentations:
            if aug == "spec_aug":
                # Frequency Masking
                mel_ref = fn.erase(
                    mel_ref,
                    anchor=fn.random.uniform(range=(0, num_mel)),
                    shape=fn.random.uniform(range=(0, num_mel // 4)),
                    normalized=False,
                    axis_names="f",
                    fill_value=fill,
                )
                # Time Masking
                mel_ref = fn.erase(
                    mel_ref,
                    anchor=fn.random.uniform(range=(0.0, 1.0)),
                    shape=fn.random.uniform(
                        range=(0, int(1 * SAMPLE_RATE / hop_length))
                    ),  # up to 1 sec
                    normalized_anchor=True,
                    normalized_shape=False,
                    axis_names="t",
                    fill_value=fill,
                )
            elif aug == "salt_pepper":
                mel_ref = fn.noise.salt_and_pepper(mel_ref, prob=0.05)
            elif aug == "white_noise":
                mel_ref = fn.noise.gaussian(mel_ref, mean=0, stddev=0.1)
            else:
                raise ValueError(f"{aug=} not recognized")
    if ref:
        return mel, mel_ref, num_frames, emodim, emo
    return mel, num_frames, emodim, emo


def benchmark_pipeline(pipe: Pipeline):
    with tqdm() as pbar:
        try:
            while True:
                _ = pipe.run()
                pbar.update(1)
        except KeyboardInterrupt:
            pass


def num_frames_to_attention_mask(num_frames, max_frames):
    batch_size = num_frames.shape[0]
    range_tensor = torch.arange(max_frames, device=num_frames.device).expand(
        batch_size, max_frames
    )
    attention_mask = (range_tensor < num_frames.unsqueeze(1)).int()
    return attention_mask
