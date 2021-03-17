import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import ConcatDataset, Dataset

from yukarin_s.config import DatasetConfig


def f0_mean(f0: numpy.ndarray, rate: float, split_second_list: List[float]):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    output = numpy.array([numpy.mean(a[a > 0]) for a in numpy.split(f0, indexes)])
    output[numpy.isnan(output)] = 0
    return output


@dataclass
class Input:
    phoneme_list: List[JvsPhoneme]
    f0: Optional[SamplingData]


@dataclass
class LazyInput:
    phoneme_list_path: SamplingData
    f0_path: Optional[Path]

    def generate(self):
        return Input(
            phoneme_list=JvsPhoneme.load_julius_list(self.phoneme_list_path),
            f0=SamplingData.load(self.f0_path) if self.f0_path is not None else None,
        )


class FeatureDataset(Dataset):
    def __init__(
        self,
        inputs: List[Union[Input, LazyInput]],
        sampling_length: int,
    ):
        self.inputs = inputs
        self.sampling_length = sampling_length

    @staticmethod
    def extract_input(
        phoneme_list_data: List[JvsPhoneme],
        f0_data: Optional[SamplingData],
        sampling_length: int,
    ):
        length = len(phoneme_list_data)

        if sampling_length > length:
            padding_length = sampling_length - length
            sampling_length = length
        else:
            padding_length = 0

        phoneme_list = numpy.array([p.phoneme_id for p in phoneme_list_data])
        phoneme_length = numpy.array([p.end - p.start for p in phoneme_list_data])

        offset = numpy.random.randint(len(phoneme_list_data) - sampling_length + 1)

        phoneme_list = phoneme_list[offset : offset + sampling_length]
        phoneme_length = phoneme_length[offset : offset + sampling_length]
        padded = numpy.zeros_like(phoneme_length, dtype=bool)

        pad_pre, pad_post = 0, 0
        if padding_length > 0:
            pad_pre = numpy.random.randint(padding_length + 1)
            pad_post = padding_length - pad_pre
            phoneme_list = numpy.pad(phoneme_list, [pad_pre, pad_post])
            phoneme_length = numpy.pad(phoneme_length, [pad_pre, pad_post])
            padded = numpy.pad(padded, [pad_pre, pad_post], constant_values=True)

        data = dict(
            phoneme_list=phoneme_list.astype(numpy.int64),
            phoneme_length=phoneme_length.astype(numpy.float32),
            padded=padded,
        )

        if f0_data is not None:
            f0 = f0_mean(
                f0=f0_data.array,
                rate=f0_data.rate,
                split_second_list=[p.end for p in phoneme_list_data[:-1]],
            )
            f0 = f0[offset : offset + sampling_length]
            if padding_length > 0:
                f0 = numpy.pad(f0, [pad_pre, pad_post])

            data["f0"] = f0.astype(numpy.float32)

        return data

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return self.extract_input(
            phoneme_list_data=input.phoneme_list,
            f0_data=input.f0,
            sampling_length=self.sampling_length,
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

    f0_paths: Optional[Dict[str, Path]] = None
    if config.f0_glob is not None:
        f0_paths = {Path(p).stem: Path(p) for p in glob(config.f0_glob)}
        assert set(fn_list) == set(f0_paths.keys())

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
                f0_path=f0_paths[fn] if f0_paths is not None else None,
            )
            for fn in fns
        ]

        dataset = FeatureDataset(inputs=inputs, sampling_length=config.sampling_length)

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
