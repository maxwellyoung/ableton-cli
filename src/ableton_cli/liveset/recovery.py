"""Find missing project samples by basename and install a reviewed file atomically."""

import os
import secrets
import shutil
import stat
import unicodedata
from contextlib import ExitStack, contextmanager
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


def _name_key(name: str) -> str:
    return unicodedata.normalize("NFC", name.casefold())


def _blocked_reason(root: Path, relative: Path, sizes: set[int]) -> str | None:
    target = root / relative
    if any(
        part.is_symlink() or part.is_junction()
        for part in (target, *target.parents)
        if part.is_relative_to(root)
    ):
        return "linked_path"
    if len(sizes) > 1:
        return "conflicting_sizes"
    try:
        info = target.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return "not_regular_file"
    return "size_mismatch" if sizes and info.st_size not in sizes else "present"


def _inventory(path: Path):
    document = load_live_set(path)
    root = find_project_root(path)
    if root is None:
        raise _invalid("No project root found.", "Choose a set inside an Ableton project.")
    missing = {}
    blocked = {}
    refs = project_sample_sizes(document)
    combined = {}
    for sample, sizes in refs.items():
        combined.setdefault(PureWindowsPath(sample).as_posix(), set()).update(sizes)
    for sample in sorted(refs):
        stored = PureWindowsPath(sample)
        sizes = combined[stored.as_posix()]
        if not stored.parts or any(
            part == ".." or PureWindowsPath(part).anchor for part in stored.parts
        ):
            reason = "unsafe_path"
        else:
            relative = Path(*stored.parts)
            try:
                reason = _blocked_reason(root, relative, sizes)
            except OSError:
                reason = "inaccessible_path"
            if reason is None:
                missing[sample] = MissingSample(relative, sizes)
                continue
        if reason != "present":
            blocked[sample] = {"sample": sample, "reason": reason, "expected_sizes": sorted(sizes)}
    return root, missing, blocked


def _io_error(exc: OSError) -> AppError:
    return AppError(
        ErrorCode.SAMPLE_IO_FAILED,
        f"Sample recovery failed: {exc.strerror}",
        "Check disk space, directory access and filesystem support.",
        ExitCode.EXECUTION_FAILED,
    )


def find_samples(path: Path, library: Path) -> dict[str, Any]:
    _, missing, blocked = _inventory(path)
    if not library.is_dir():
        raise _invalid("Sample library directory not found.", "Pass a directory with --library.")
    index: dict[str, list[tuple[str, int]]] = {_name_key(p.path.name): [] for p in missing.values()}

    def fail(exc: OSError) -> None:
        raise _io_error(exc) from exc

    if not index:
        return {"missing": [], "blocked": list(blocked.values())}
    try:
        for directory, _, files in library.walk(on_error=fail):
            for name in files:
                candidates = index.get(_name_key(name))
                if candidates is None:
                    continue
                file = directory / name
                metadata = file.lstat()
                if stat.S_ISREG(metadata.st_mode):
                    candidates.append((file.relative_to(library).as_posix(), metadata.st_size))
    except OSError as exc:
        raise _io_error(exc) from exc
    return {
        "blocked": list(blocked.values()),
        "missing": [
            {
                "sample": sample,
                "candidates": sorted(
                    name
                    for name, size in index[_name_key(target.path.name)]
                    if target.accepts(size)
                ),
            }
            for sample, target in missing.items()
        ],
    }


@contextmanager
def _destination_parent(root: Path, relative: Path):
    required = {os.open, os.mkdir, os.rmdir, os.unlink, os.link}
    if not required <= os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise AppError(
            ErrorCode.SAMPLE_IO_FAILED,
            "Safe directory-relative restore unavailable.",
            "Restore on a POSIX filesystem supporting directory handles and hard links.",
            ExitCode.EXECUTION_FAILED,
        )
    flags = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_RDONLY
    with ExitStack() as stack:

        def pin(name, parent=None):
            fd = os.open(name, flags, dir_fd=parent)
            stack.callback(os.close, fd)
            return fd

        parent = pin(root.anchor)
        for part in root.parts[1:]:
            parent = pin(part, parent)
        for part in relative.parts:
            try:
                child = pin(part, parent)
            except FileNotFoundError:
                try:
                    os.mkdir(part, dir_fd=parent)
                except FileExistsError:
                    pass  # Another restore may have created the directory.
                child = pin(part, parent)
            parent = child
        yield parent


@contextmanager
def _staging_directory(parent: int):
    # A shared project directory must not expose the staged file by name.
    name = ".restore-" + secrets.token_hex(16)
    os.mkdir(name, mode=0o700, dir_fd=parent)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise _invalid(
                    "Cannot create private staging directory.", "Check project permissions."
                )
            yield fd
        finally:
            os.close(fd)
    finally:
        os.rmdir(name, dir_fd=parent)


def _copy_sample(parent: int, staging: int, entry: MissingSample, source: Path) -> None:
    temporary = "sample"
    with source.open("rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not entry.accepts(metadata.st_size):
            raise _invalid(
                "Source type or size differs from the set reference.", "Select a matching file."
            )
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            metadata.st_mode & 0o666,
            dir_fd=staging,
        )
        try:
            with os.fdopen(fd, "wb") as output:
                shutil.copyfileobj(handle, output)
                if output.tell() != metadata.st_size:
                    raise _invalid("Source size changed during restore.", "Recheck the source.")
            os.link(
                temporary,
                entry.path.name,
                src_dir_fd=staging,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
        finally:
            os.unlink(temporary, dir_fd=staging)


def restore_sample(path: Path, sample: str, source: Path) -> dict[str, str]:
    root, missing, blocked = _inventory(path)
    if sample in blocked:
        raise _invalid(
            f"Blocked sample: {blocked[sample]['reason']}", "Inspect set find-samples for details."
        )
    entry = missing.get(sample)
    if entry is None:
        raise _invalid("Sample is not a missing project reference.", "Run set find-samples again.")
    target = entry.path
    if not source.is_file() or _name_key(source.name) != _name_key(target.name):
        raise _invalid("Source must be a file with the expected name.", "Select a listed match.")
    try:
        with _destination_parent(root, target.parent) as parent:
            with _staging_directory(parent) as staging:
                _copy_sample(parent, staging, entry, source)
    except FileExistsError as exc:
        raise _invalid("Destination already exists.", "Run set find-samples again.") from exc
    except OSError as exc:
        raise _io_error(exc) from exc
    return {"restored": sample}
