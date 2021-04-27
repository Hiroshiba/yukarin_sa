import json
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import ConcatDataset, Dataset

from yukarin_sa.config import DatasetConfig

unvoiced_mora_phoneme_list = ["A", "I", "U", "E", "O", "cl", "pau"]
mora_phoneme_list = ["a", "i", "u", "e", "o", "N"] + unvoiced_mora_phoneme_list
voiced_phoneme_list = (
    ["a", "i", "u", "e", "o", "N"]
    + ["n", "m", "y", "r", "w", "g", "z", "j", "d", "b"]
    + ["ny", "my", "ry", "gy", "by"]
)


class F0ProcessMode(str, Enum):
    phoneme = "phoneme"
    # mora_vowel = "mora_vowel"
    mora = "mora"
    voiced_mora = "voiced_mora"


def f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: List[float],
    weight: Optional[numpy.ndarray],
):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    if weight is None:
        output = numpy.array([numpy.mean(a[a > 0]) for a in numpy.split(f0, indexes)])
    else:
        output = numpy.array(
            [
                numpy.sum(a[a > 0] * b[a > 0]) / numpy.sum(b[a > 0])
                for a, b in zip(numpy.split(f0, indexes), numpy.split(weight, indexes))
            ]
        )
    output[numpy.isnan(output)] = 0
    return output


def split_mora(phoneme_list: List[JvsPhoneme]):
    vowel_indexes = [
        i for i, p in enumerate(phoneme_list) if p.phoneme in mora_phoneme_list
    ]
    vowel_phoneme_list = [phoneme_list[i] for i in vowel_indexes]
    consonant_phoneme_list: List[Optional[JvsPhoneme]] = [None] + [
        None if post - prev == 1 else phoneme_list[post - 1]
        for prev, post in zip(vowel_indexes[:-1], vowel_indexes[1:])
    ]
    return consonant_phoneme_list, vowel_phoneme_list, vowel_indexes


@dataclass
class Input:
    phoneme_list: List[JvsPhoneme]
    start_accent_list: numpy.ndarray
    end_accent_list: numpy.ndarray
    start_accent_phrase_list: numpy.ndarray
    end_accent_phrase_list: numpy.ndarray
    f0: SamplingData
    volume: Optional[SamplingData]


@dataclass
class LazyInput:
    phoneme_list_path: Path
    start_accent_list_path: Path
    end_accent_list_path: Path
    start_accent_phrase_list_path: Path
    end_accent_phrase_list_path: Path
    f0_path: Path
    volume_path: Optional[Path]

    def generate(self):
        return Input(
            phoneme_list=JvsPhoneme.load_julius_list(self.phoneme_list_path),
            start_accent_list=numpy.array(
                [bool(int(s)) for s in self.start_accent_list_path.read_text().split()]
            ),
            end_accent_list=numpy.array(
                [bool(int(s)) for s in self.end_accent_list_path.read_text().split()]
            ),
            start_accent_phrase_list=numpy.array(
                [
                    bool(int(s))
                    for s in self.start_accent_phrase_list_path.read_text().split()
                ]
            ),
            end_accent_phrase_list=numpy.array(
                [
                    bool(int(s))
                    for s in self.end_accent_phrase_list_path.read_text().split()
                ]
            ),
            f0=SamplingData.load(self.f0_path),
            volume=(
                SamplingData.load(self.volume_path)
                if self.volume_path is not None
                else None
            ),
        )


class FeatureDataset(Dataset):
    def __init__(
        self,
        inputs: List[Union[Input, LazyInput]],
        sampling_length: int,
        f0_process_mode: F0ProcessMode,
        phoneme_mask_max_length: int,
        phoneme_mask_num: int,
        accent_mask_max_length: int,
        accent_mask_num: int,
    ):
        self.inputs = inputs
        self.sampling_length = sampling_length
        self.f0_process_mode = f0_process_mode
        self.phoneme_mask_max_length = phoneme_mask_max_length
        self.phoneme_mask_num = phoneme_mask_num
        self.accent_mask_max_length = accent_mask_max_length
        self.accent_mask_num = accent_mask_num

    @staticmethod
    def extract_input(
        phoneme_list_data: List[JvsPhoneme],
        start_accent_list: numpy.ndarray,
        end_accent_list: numpy.ndarray,
        start_accent_phrase_list: numpy.ndarray,
        end_accent_phrase_list: numpy.ndarray,
        f0_data: SamplingData,
        volume_data: Optional[SamplingData],
        sampling_length: int,
        f0_process_mode: F0ProcessMode,
        phoneme_mask_max_length: int,
        phoneme_mask_num: int,
        accent_mask_max_length: int,
        accent_mask_num: int,
    ):
        (
            consonant_phoneme_list_data,
            vowel_phoneme_list_data,
            vowel_indexes,
        ) = split_mora(phoneme_list_data)

        start_accent_list = start_accent_list[vowel_indexes]
        end_accent_list = end_accent_list[vowel_indexes]
        start_accent_phrase_list = start_accent_phrase_list[vowel_indexes]
        end_accent_phrase_list = end_accent_phrase_list[vowel_indexes]

        rate = f0_data.rate
        f0_array = f0_data.array

        if volume_data is not None:
            volume_array = volume_data.resample(rate)

            min_length = min(len(f0_array), len(volume_array))
            f0_array = f0_array[:min_length]
            volume_array = volume_array[:min_length]
        else:
            volume_array = None

        f0 = f0_mean(
            f0=f0_array,
            rate=rate,
            split_second_list=[p.end for p in phoneme_list_data[:-1]],
            weight=volume_array,
        )

        if f0_process_mode == F0ProcessMode.phoneme:
            consonant_phoneme_list_data = None
        elif f0_process_mode == F0ProcessMode.mora:
            f0 = f0[vowel_indexes]
            phoneme_list_data = vowel_phoneme_list_data
        elif f0_process_mode == F0ProcessMode.voiced_mora:
            phoneme_length_list = [p.end - p.start for p in phoneme_list_data]
            new_f0 = []
            for i, d in enumerate(numpy.diff(numpy.r_[0, vowel_indexes])):
                index = vowel_indexes[i]
                if (
                    d == 1
                    or phoneme_list_data[index - 1].phoneme not in voiced_phoneme_list
                ):
                    new_f0.append(f0[index])
                else:
                    a = f0[index - 1] * phoneme_length_list[index - 1]
                    b = f0[index] * phoneme_length_list[index]
                    new_f0.append(
                        (a + b)
                        / (phoneme_length_list[index] + phoneme_length_list[index - 1])
                    )
            f0 = numpy.array(new_f0)
            phoneme_list_data = vowel_phoneme_list_data

        consonant_phoneme_list: Optional[numpy.ndarray] = None
        if consonant_phoneme_list_data is not None:
            consonant_phoneme_list = numpy.array(
                [
                    p.phoneme_id if p is not None else -1
                    for p in consonant_phoneme_list_data
                ]
            )

        phoneme_list = numpy.array([p.phoneme_id for p in phoneme_list_data])

        # voiced / unvoiced
        voiced = numpy.array(
            [p.phoneme in voiced_phoneme_list for p in phoneme_list_data]
        )
        f0[~voiced] = 0

        length = len(phoneme_list_data)

        if sampling_length > length:
            padding_length = sampling_length - length
            sampling_length = length
        else:
            padding_length = 0

        offset = numpy.random.randint(len(phoneme_list_data) - sampling_length + 1)
        offset_slice = slice(offset, offset + sampling_length)
        phoneme_list = phoneme_list[offset_slice]
        start_accent_list = start_accent_list[offset_slice]
        end_accent_list = end_accent_list[offset_slice]
        start_accent_phrase_list = start_accent_phrase_list[offset_slice]
        end_accent_phrase_list = end_accent_phrase_list[offset_slice]
        f0 = f0[offset_slice]
        voiced = voiced[offset_slice]
        padded = numpy.zeros_like(phoneme_list, dtype=bool)
        if consonant_phoneme_list is not None:
            consonant_phoneme_list = consonant_phoneme_list[offset_slice]

        pad_pre, pad_post = 0, 0
        if padding_length > 0:
            pad_pre = numpy.random.randint(padding_length + 1)
            pad_post = padding_length - pad_pre
            pad_list = [pad_pre, pad_post]
            phoneme_list = numpy.pad(phoneme_list, pad_list)
            start_accent_list = numpy.pad(start_accent_list, pad_list)
            end_accent_list = numpy.pad(end_accent_list, pad_list)
            start_accent_phrase_list = numpy.pad(start_accent_phrase_list, pad_list)
            end_accent_phrase_list = numpy.pad(end_accent_phrase_list, pad_list)
            f0 = numpy.pad(f0, pad_list)
            voiced = numpy.pad(voiced, pad_list)
            padded = numpy.pad(padded, pad_list, constant_values=True)
            if consonant_phoneme_list is not None:
                consonant_phoneme_list = numpy.pad(consonant_phoneme_list, pad_list)

        if phoneme_mask_max_length > 0 and phoneme_mask_num > 0:
            for _ in range(phoneme_mask_num):
                mask_length = numpy.random.randint(phoneme_mask_max_length)
                mask_offset = numpy.random.randint(len(phoneme_list) - mask_length + 1)
                phoneme_list[mask_offset : mask_offset + mask_length] = -1
                if consonant_phoneme_list is not None:
                    consonant_phoneme_list[mask_offset : mask_offset + mask_length] = -1

        if accent_mask_max_length > 0 and accent_mask_num > 0:
            for _ in range(accent_mask_num):
                mask_length = numpy.random.randint(accent_mask_max_length)
                mask_offset = numpy.random.randint(
                    len(start_accent_list) - mask_length + 1
                )
                start_accent_list[mask_offset : mask_offset + mask_length] = 0
                end_accent_list[mask_offset : mask_offset + mask_length] = 0
                start_accent_phrase_list[mask_offset : mask_offset + mask_length] = 0
                end_accent_phrase_list[mask_offset : mask_offset + mask_length] = 0

        data = dict(
            phoneme_list=phoneme_list.astype(numpy.int64),
            start_accent_list=start_accent_list.astype(numpy.int64),
            end_accent_list=end_accent_list.astype(numpy.int64),
            start_accent_phrase_list=start_accent_phrase_list.astype(numpy.int64),
            end_accent_phrase_list=end_accent_phrase_list.astype(numpy.int64),
            f0=f0.astype(numpy.float32),
            voiced=voiced,
            padded=padded,
        )
        if consonant_phoneme_list is not None:
            data["consonant_phoneme_list"] = consonant_phoneme_list.astype(numpy.int64)
        return data

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return self.extract_input(
            phoneme_list_data=input.phoneme_list,
            start_accent_list=input.start_accent_list,
            end_accent_list=input.end_accent_list,
            start_accent_phrase_list=input.start_accent_phrase_list,
            end_accent_phrase_list=input.end_accent_phrase_list,
            f0_data=input.f0,
            volume_data=input.volume,
            sampling_length=self.sampling_length,
            f0_process_mode=self.f0_process_mode,
            phoneme_mask_max_length=self.phoneme_mask_max_length,
            phoneme_mask_num=self.phoneme_mask_num,
            accent_mask_max_length=self.accent_mask_max_length,
            accent_mask_num=self.accent_mask_num,
        )


class SpeakerFeatureDataset(Dataset):
    def __init__(self, dataset: FeatureDataset, speaker_ids: List[int]):
        assert len(dataset) == len(speaker_ids)
        self.dataset = dataset
        self.speaker_ids = speaker_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        d = self.dataset[i]
        d["speaker_id"] = numpy.array(self.speaker_ids[i], dtype=numpy.int64)
        return d


class TensorWrapperDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return default_convert(self.dataset[i])


def create_dataset(config: DatasetConfig):
    phoneme_list_paths = {Path(p).stem: Path(p) for p in glob(config.phoneme_list_glob)}
    fn_list = sorted(phoneme_list_paths.keys())
    assert len(fn_list) > 0

    start_accent_list_paths = {
        Path(p).stem: Path(p) for p in glob(config.start_accent_list_glob)
    }
    assert set(fn_list) == set(start_accent_list_paths.keys())

    end_accent_list_paths = {
        Path(p).stem: Path(p) for p in glob(config.end_accent_list_glob)
    }
    assert set(fn_list) == set(end_accent_list_paths.keys())

    start_accent_phrase_list_paths = {
        Path(p).stem: Path(p) for p in glob(config.start_accent_phrase_list_glob)
    }
    assert set(fn_list) == set(start_accent_phrase_list_paths.keys())

    end_accent_phrase_list_paths = {
        Path(p).stem: Path(p) for p in glob(config.end_accent_phrase_list_glob)
    }
    assert set(fn_list) == set(end_accent_phrase_list_paths.keys())

    f0_paths = {Path(p).stem: Path(p) for p in glob(config.f0_glob)}
    assert set(fn_list) == set(f0_paths.keys())

    volume_paths = {Path(p).stem: Path(p) for p in glob(config.volume_glob)}
    assert set(fn_list) == set(volume_paths.keys())

    speaker_ids: Optional[Dict[str, int]] = None
    if config.speaker_dict_path is not None:
        fn_each_speaker: Dict[str, List[str]] = json.loads(
            config.speaker_dict_path.read_text()
        )
        assert config.speaker_size == len(fn_each_speaker)

        speaker_ids = {
            fn: speaker_id
            for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_ids.keys()))

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    test_num = config.test_num
    trains = fn_list[test_num:]
    tests = fn_list[:test_num]

    def _dataset(fns, for_test=False):
        inputs = [
            LazyInput(
                phoneme_list_path=phoneme_list_paths[fn],
                start_accent_list_path=start_accent_list_paths[fn],
                end_accent_list_path=end_accent_list_paths[fn],
                start_accent_phrase_list_path=start_accent_phrase_list_paths[fn],
                end_accent_phrase_list_path=end_accent_phrase_list_paths[fn],
                f0_path=f0_paths[fn],
                volume_path=volume_paths[fn],
            )
            for fn in fns
        ]

        if not for_test:
            dataset = FeatureDataset(
                inputs=inputs,
                sampling_length=config.sampling_length,
                f0_process_mode=F0ProcessMode(config.f0_process_mode),
                phoneme_mask_max_length=config.phoneme_mask_max_length,
                phoneme_mask_num=config.phoneme_mask_num,
                accent_mask_max_length=config.accent_mask_max_length,
                accent_mask_num=config.accent_mask_num,
            )
        else:
            dataset = FeatureDataset(
                inputs=inputs,
                sampling_length=config.sampling_length,
                f0_process_mode=F0ProcessMode(config.f0_process_mode),
                phoneme_mask_max_length=0,
                phoneme_mask_num=0,
                accent_mask_max_length=0,
                accent_mask_num=0,
            )

        if speaker_ids is not None:
            dataset = SpeakerFeatureDataset(
                dataset=dataset,
                speaker_ids=[speaker_ids[fn] for fn in fns],
            )

        dataset = TensorWrapperDataset(dataset)

        if for_test:
            dataset = ConcatDataset([dataset] * config.test_trial_num)

        return dataset

    return {
        "train": _dataset(trains),
        "test": _dataset(tests, for_test=True),
    }
