"""Find missing project samples by basename and install a reviewed file atomically."""

import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from ..errors import AppError, ErrorCode, ExitCode
from .document import load_live_set
from .media import find_project_root, project_sample_sizes


def _invalid(message: str, hint: str) -> AppError:
    return AppError(ErrorCode.INVALID_ARGUMENT, message, hint, ExitCode.INVALID_ARGUMENT)


@dataclass(frozen=True)
class MissingSample:
    path: Path
    sizes: set[int]

    def accepts(self, size: int) -> bool:
        return not self.sizes or self.sizes == {size}


def _missing(path: Path) -> dict[str, MissingSample]:
    document = load_live_set(path)
    root = find_project_root(path)
    if root is None:
        raise _invalid("No project root found.", "Choose a set inside an Ableton project.")
    missing = {}
    for sample, sizes in sorted(project_sample_sizes(document).items()):
        relative = Path(sample)
        if PureWindowsPath(sample).anchor or ".." in PureWindowsPath(sample).parts:
            raise _invalid("Unsafe project sample path.", "Inspect project references in Live.")
        target = root / relative
        if not target.resolve().is_relative_to(root):
            raise _invalid("Sample points outside the project.", "Inspect project links in Live.")
        if not target.is_file():
            missing[sample] = MissingSample(target, sizes)
    return missing


def _io_error(exc: OSError) -> AppError:
    return AppError(
        ErrorCode.SAMPLE_IO_FAILED,
        f"Sample recovery failed: {exc.strerror}",
        "Check disk space, directory access and filesystem support.",
        ExitCode.EXECUTION_FAILED,
    )


def find_samples(path: Path, library: Path) -> dict[str, Any]:
    missing = _missing(path)
    if not library.is_dir():
        raise _invalid("Sample library directory not found.", "Pass a directory with --library.")
    index: dict[str, list[tuple[str, int]]] = {p.path.name.casefold(): [] for p in missing.values()}

    def fail(exc: OSError) -> None:
        raise _io_error(exc) from exc

    if not index:
        return {"missing": []}
    try:
        for directory, _, files in library.walk(on_error=fail):
            for name in files:
                candidates = index.get(name.casefold())
                if candidates is None:
                    continue
                file = directory / name
                metadata = file.lstat()
                if stat.S_ISREG(metadata.st_mode):
                    candidates.append((file.relative_to(library).as_posix(), metadata.st_size))
    except OSError as exc:
        raise _io_error(exc) from exc
    return {
        "missing": [
            {
                "sample": sample,
                "candidates": sorted(
                    name
                    for name, size in index[target.path.name.casefold()]
                    if target.accepts(size)
                ),
            }
            for sample, target in missing.items()
        ]
    }


def restore_sample(path: Path, sample: str, source: Path) -> dict[str, str]:
    missing = _missing(path)
    entry = missing.get(sample)
    if entry is None:
        raise _invalid("Sample is not a missing project reference.", "Run set find-samples again.")
    target = entry.path
    if not source.is_file() or source.name.casefold() != target.name.casefold():
        raise _invalid("Source must be a file with the expected name.", "Select a listed match.")
    try:
        if not entry.accepts(source.stat().st_size):
            raise _invalid("File size differs from the set reference.", "Select a matching file.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete_on_close=False) as temp:
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, temp)
            if not entry.accepts(temp.tell()):
                raise _invalid("Copied size differs from the set reference.", "Recheck the source.")
            temp.close()
            os.link(temp.name, target)
    except FileExistsError as exc:
        raise _invalid("Destination already exists.", "Run set find-samples again.") from exc
    except OSError as exc:
        raise _io_error(exc) from exc
    return {"restored": sample}
