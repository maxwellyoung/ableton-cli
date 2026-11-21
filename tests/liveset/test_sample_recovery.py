import json
import os

import pytest
from live_set_builder import SampleSpec, SetSpec, TrackSpec


def test_recover_missing_samples(runner, cli_app, live_set_factory, tmp_path):
    samples = [SampleSpec(relative_path=f"Samples/{name}") for name in ("vox.wav", "bd.wav")]
    path = live_set_factory(SetSpec(tracks=[TrackSpec(tag="AudioTrack", samples=samples)]))
    (path.parent / "Ableton Project Info").mkdir()
    original = path.read_bytes()
    source = tmp_path / "library"
    source.mkdir()
    (source / "vox.wav").write_bytes(b"take 1")
    (source / "alt").mkdir()
    (source / "alt" / "vox.wav").write_bytes(b"take 2")
    args = ["--output", "json", "set"]
    search = runner.invoke(
        cli_app, [*args, "find-samples", "--path", str(path), "--library", str(source)]
    )
    assert search.exit_code == 0, search.output
    assert json.loads(search.stdout)["result"]["missing"] == [
        {"sample": "Samples/bd.wav", "candidates": []},
        {"sample": "Samples/vox.wav", "candidates": ["alt/vox.wav", "vox.wav"]},
    ]
    restore = [
        *args,
        "restore-sample",
        "--path",
        str(path),
        "--sample",
        "Samples/vox.wav",
        "--source",
        str(source / "alt/vox.wav"),
    ]
    assert runner.invoke(cli_app, ["--read-only", *restore]).exit_code != 0
    if os.name == "nt":
        assert runner.invoke(cli_app, restore).exit_code == 20
        return
    assert runner.invoke(cli_app, restore).exit_code == 0
    assert (path.parent / "Samples/vox.wav").read_bytes() == b"take 2"
    assert path.read_bytes() == original
    assert runner.invoke(cli_app, restore).exit_code == 2
    assert runner.invoke(cli_app, [*restore[:7], "../outside.wav", *restore[8:]]).exit_code != 0


def test_refuses_reference_outside_project(runner, cli_app, live_set_factory):
    samples = [SampleSpec(relative_path="../outside.wav")]
    path = live_set_factory(SetSpec(tracks=[TrackSpec(tag="AudioTrack", samples=samples)]))
    (path.parent / "Ableton Project Info").mkdir()
    result = runner.invoke(
        cli_app,
        [
            "--output",
            "json",
            "set",
            "find-samples",
            "--path",
            str(path),
            "--library",
            str(path.parent),
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["result"]["blocked"][0]["reason"] == "unsafe_path"


@pytest.mark.skipif(os.name != "posix", reason="requires directory-relative writes")
def test_interrupted_copy_leaves_no_partial_sample(live_set_factory, tmp_path, monkeypatch):

    from ableton_cli.errors import AppError
    from ableton_cli.liveset import recovery

    path = live_set_factory(
        SetSpec(
            tracks=[
                TrackSpec(tag="AudioTrack", samples=[SampleSpec(relative_path="Samples/kick.wav")])
            ]
        )
    )
    (path.parent / "Ableton Project Info").mkdir()
    source = tmp_path / "kick.wav"
    source.write_bytes(b"audio")

    def interrupt(handle, temp):
        temp.write(handle.read(1))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(recovery.shutil, "copyfileobj", interrupt)
    with pytest.raises(AppError, match="No space left"):
        recovery.restore_sample(path, "Samples/kick.wav", source)
    assert list((path.parent / "Samples").iterdir()) == []


def test_recorded_size_excludes_wrong_take(runner, cli_app, live_set_factory, tmp_path):
    import xml.etree.ElementTree as ET

    def add_size(root):
        ET.SubElement(root.find(".//SampleRef/FileRef"), "OriginalFileSize", Value="4")

    path = live_set_factory(
        SetSpec(
            tracks=[
                TrackSpec(tag="AudioTrack", samples=[SampleSpec(relative_path="Samples/take.wav")])
            ]
        ),
        mutate=add_size,
    )
    (path.parent / "Ableton Project Info").mkdir()
    source = tmp_path / "take.wav"
    source.write_bytes(b"wrong")
    args = ["--output", "json", "set"]
    result = runner.invoke(
        cli_app, [*args, "find-samples", "--path", str(path), "--library", str(tmp_path)]
    )
    assert json.loads(result.stdout)["result"]["missing"][0]["candidates"] == []
    result = runner.invoke(
        cli_app,
        [
            *args,
            "restore-sample",
            "--path",
            str(path),
            "--sample",
            "Samples/take.wav",
            "--source",
            str(source),
        ],
    )
    assert result.exit_code == (20 if os.name == "nt" else 2)


def test_portable_names_and_blocked_refs(runner, cli_app, live_set_factory, tmp_path):
    import unicodedata

    if os.name != "posix":
        pytest.skip("POSIX directory handle restore")
    samples = [
        SampleSpec(relative_path=name)
        for name in (r"Samples\Café.wav", "../outside.wav", "Linked/Other.wav")
    ]
    path = live_set_factory(SetSpec(tracks=[TrackSpec(tag="AudioTrack", samples=samples)]))
    (tmp_path / "Ableton Project Info").mkdir()
    library = tmp_path / "library"
    library.mkdir()
    (tmp_path / "Linked").symlink_to(library, target_is_directory=True)
    source = library / unicodedata.normalize("NFD", "Café.wav")
    source.write_bytes(b"sample")
    source.chmod(0o644)
    args = ["--output", "json", "set"]
    search = runner.invoke(
        cli_app, [*args, "find-samples", "--path", str(path), "--library", str(library)]
    )
    assert search.exit_code == 0, search.output
    result = json.loads(search.stdout)["result"]
    assert len(result["blocked"]) == 2
    assert result["missing"] == [{"sample": samples[0].relative_path, "candidates": [source.name]}]
    restore = runner.invoke(
        cli_app,
        [
            *args,
            "restore-sample",
            "--path",
            str(path),
            "--sample",
            samples[0].relative_path,
            "--source",
            str(source),
        ],
    )
    assert restore.exit_code == 0, restore.output
    target = tmp_path / "Samples/Café.wav"
    assert target.read_bytes() == b"sample"
    reference = tmp_path / "normal-mode"
    reference.touch(mode=0o644)
    assert target.stat().st_mode & 0o777 == reference.stat().st_mode & 0o777
    lint = runner.invoke(cli_app, [*args, "lint", "--path", str(path)])
    assert lint.exit_code == 0, lint.stdout
    findings = json.loads(lint.stdout)["result"]["findings"]
    missing = [
        item["relative_path"]
        for finding in findings
        if finding["code"] == "media_missing"
        for item in finding["details"]["files"]
    ]
    assert samples[0].relative_path not in missing


def test_conflicting_and_present_wrong_size(runner, cli_app, live_set_factory, tmp_path):
    import xml.etree.ElementTree as ET

    def sizes(root):
        for ref, size in zip(root.findall(".//SampleRef/FileRef"), [4, 8, 4], strict=True):
            ET.SubElement(ref, "OriginalFileSize", Value=str(size))

    samples = [SampleSpec(relative_path=name) for name in ("a.wav", "a.wav", "b.wav")]
    path = live_set_factory(
        SetSpec(tracks=[TrackSpec(tag="AudioTrack", samples=samples)]), mutate=sizes
    )
    (tmp_path / "Ableton Project Info").mkdir()
    (tmp_path / "b.wav").touch()
    result = runner.invoke(
        cli_app,
        [
            "--output",
            "json",
            "set",
            "find-samples",
            "--path",
            str(path),
            "--library",
            str(tmp_path),
        ],
    )
    assert json.loads(result.stdout)["result"]["blocked"] == [
        {"sample": "a.wav", "reason": "conflicting_sizes", "expected_sizes": [4, 8]},
        {"sample": "b.wav", "reason": "size_mismatch", "expected_sizes": [4]},
    ]


def test_restore_without_directory_handles_fails_before_writes(
    live_set_factory, tmp_path, monkeypatch
):
    from ableton_cli.errors import AppError
    from ableton_cli.liveset.recovery import restore_sample

    sample = "Samples/take.wav"
    path = live_set_factory(
        SetSpec(tracks=[TrackSpec(tag="AudioTrack", samples=[SampleSpec(relative_path=sample)])])
    )
    (tmp_path / "Ableton Project Info").mkdir()
    source = tmp_path / "take.wav"
    source.touch()
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(AppError, match="directory-relative"):
        restore_sample(path, sample, source)
    assert not (tmp_path / "Samples").exists()
