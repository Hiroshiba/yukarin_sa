from os import PathLike, replace
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import Optional


class CachePath(PathLike):
    """
    ファイルキャッシュを作る。
    `open()`か`str()`か`create_cache()`を呼び出すとキャッシュが作られる。
    ファイルが存在する場合はアクセス日時が更新される。
    """

    def __init__(
        self,
        src_path: PathLike,
        dst_path: Optional[Path] = None,
        cache_dir: Path = Path(gettempdir()),
    ):
        src_path = Path(src_path)
        if dst_path is None:
            dst_path = cache_dir.joinpath(*src_path.absolute().parts[1:])

        self.src_path = src_path
        self.dst_path = dst_path

    def create_cache(self):
        if self.dst_path.exists():
            self.dst_path.touch()
            return

        if not self.src_path.exists() or self.src_path.is_dir():
            raise FileNotFoundError(f"ファイルが存在しません: {self.src_path}")

        self.dst_path.parent.mkdir(parents=True, exist_ok=True)

        with NamedTemporaryFile(dir=str(self.dst_path.parent), delete=False) as f:
            name = f.name

        Path(name).write_bytes(self.src_path.read_bytes())
        replace(name, str(self.dst_path))

    def __str__(self):
        self.create_cache()
        return str(self.dst_path)

    def __fspath__(self):
        return str(self)
