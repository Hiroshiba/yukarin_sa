from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from yukarin_sa.utility import dataclass_utility
from yukarin_sa.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    root_dir: Path
    phoneme_list_pathlist_path: Path
    start_accent_list_pathlist_path: Path
    end_accent_list_pathlist_path: Path
    start_accent_phrase_list_pathlist_path: Path
    end_accent_phrase_list_pathlist_path: Path
    f0_pathlist_path: Path
    volume_pathlist_path: Path
    sampling_length: int
    f0_process_mode: str
    phoneme_type: str
    phoneme_mask_max_length: int
    phoneme_mask_num: int
    accent_mask_max_length: int
    accent_mask_num: int
    f0_mask_max_length: int
    f0_mask_num: int
    speaker_dict_path: Optional[Path]
    speaker_size: Optional[int]
    test_num: int
    test_trial_num: int = 1
    valid_phoneme_list_pathlist_path: Optional[Path] = None
    valid_start_accent_list_pathlist_path: Optional[Path] = None
    valid_end_accent_list_pathlist_path: Optional[Path] = None
    valid_start_accent_phrase_list_pathlist_path: Optional[Path] = None
    valid_end_accent_phrase_list_pathlist_path: Optional[Path] = None
    valid_f0_pathlist_path: Optional[Path] = None
    valid_volume_pathlist_path: Optional[Path] = None
    valid_speaker_dict_path: Optional[Path] = None
    valid_trial_num: Optional[int] = None
    valid_num: Optional[int] = None
    seed: int = 0


@dataclass
class NetworkConfig:
    phoneme_size: int
    phoneme_embedding_size: int
    encoder_type: str
    encoder_hidden_size: int
    encoder_kernel_size: int
    encoder_layer_num: int
    ar_encoder_type: Optional[str]
    ar_encoder_hidden_size: int
    ar_encoder_kernel_size: int
    ar_encoder_layer_num: int
    speaker_size: int
    speaker_embedding_size: int


@dataclass
class ModelConfig:
    f0_loss_weight: float


@dataclass
class TrainConfig:
    batch_size: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    weight_initializer: Optional[str] = None
    pretrained_predictor_path: Optional[Path] = None
    step_shift: Optional[Dict[str, Any]] = None
    num_processes: Optional[int] = None
    use_gpu: bool = True
    use_amp: bool = False
    use_multithread: bool = False


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    if "f0_process_mode" not in d["dataset"]:
        d["dataset"]["f0_process_mode"] = "phoneme"

    if "f0_mask_max_length" not in d["dataset"]:
        d["dataset"]["f0_mask_max_length"] = 0
    if "f0_mask_num" not in d["dataset"]:
        d["dataset"]["f0_mask_num"] = 0

    if "phoneme_type" not in d["dataset"]:
        d["dataset"]["phoneme_type"] = "jvs"

    if "root_dir" not in d["dataset"]:
        d["dataset"]["root_dir"] = Path(".")
    for before_key in [
        "phoneme_list_glob",
        "start_accent_list_glob",
        "end_accent_list_glob",
        "start_accent_phrase_list_glob",
        "end_accent_phrase_list_glob",
        "f0_glob",
        "volume_glob",
        "valid_phoneme_list_glob",
        "valid_start_accent_list_glob",
        "valid_end_accent_list_glob",
        "valid_start_accent_phrase_list_glob",
        "valid_end_accent_phrase_list_glob",
        "valid_f0_glob",
        "valid_volume_glob",
    ]:
        if before_key in d["dataset"]:
            after_key = before_key.replace("_glob", "_pathlist_path")
            d["dataset"][after_key] = d["dataset"].pop(before_key)
