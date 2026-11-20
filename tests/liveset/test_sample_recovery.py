import json

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
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "INVALID_ARGUMENT"


def test_interrupted_copy_leaves_no_partial_sample(live_set_factory, tmp_path, monkeypatch):
    import pytest

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
    assert result.exit_code == 2
