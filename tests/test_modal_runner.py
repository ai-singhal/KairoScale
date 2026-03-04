"""Tests for Modal sandbox runner compatibility behavior."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from KairoScale.sandbox import modal_runner


class _FakeImage:
    @classmethod
    def from_registry(cls, *_args, **_kwargs):
        return cls()

    def pip_install(self, *_args, **_kwargs):
        return self

    def add_local_dir(self, *_args, **_kwargs):
        return self

    def add_local_file(self, *_args, **_kwargs):
        return self


class _FakeFileHandle:
    def __init__(self, payload):
        self._payload = payload
        self.read = SimpleNamespace(aio=self._read)
        self.close = SimpleNamespace(aio=self._close)

    async def _read(self, _n=None):
        return self._payload

    async def _close(self):
        return None


class _FakeStream:
    def __init__(self, payload: str):
        self._payload = payload
        self.read = SimpleNamespace(aio=self._read)

    async def _read(self, _n=None):
        return self._payload


class _FakeEntry:
    def __init__(self, path: str):
        self.path = path


class _FakeProcess:
    def __init__(self):
        self.wait = SimpleNamespace(aio=self._wait)

    async def _wait(self):
        return 0


class _FakeSandbox:
    def __init__(self):
        self.object_id = "sb-123"
        self.stdout = _FakeStream("hello from sandbox\n")
        self.stderr = _FakeStream("")
        self._files = {
            "/tmp/KairoScale_wrapper_exit_code": "0\n",
            "/tmp/KairoScale_artifacts/profile_result.json": b'{"ok": true}',
            "/tmp/KairoScale_artifacts/metrics.json": b'{"steps": 3}',
        }
        self.ls = SimpleNamespace(aio=self._ls)
        self.open = SimpleNamespace(aio=self._open)
        self.exec = SimpleNamespace(aio=self._exec)
        self.terminate = SimpleNamespace(aio=self._terminate)
        self.wait = SimpleNamespace(aio=self._wait)

    async def _ls(self, path: str):
        if path == "/tmp":
            async def _tmp_gen():
                yield "KairoScale_wrapper_done"
                yield "KairoScale_wrapper_exit_code"
            return _tmp_gen()
        if path == "/tmp/KairoScale_artifacts":
            async def _artifact_gen():
                yield _FakeEntry("/tmp/KairoScale_artifacts/profile_result.json")
                yield _FakeEntry("/tmp/KairoScale_artifacts/metrics.json")
            return _artifact_gen()
        return []

    async def _open(self, path: str, _mode: str = "r"):
        if path not in self._files:
            raise FileNotFoundError(path)
        return _FakeFileHandle(self._files[path])

    async def _exec(self, *_args, **_kwargs):
        return _FakeProcess()

    async def _terminate(self, **_kwargs):
        return 0

    async def _wait(self, **_kwargs):
        return 0


@pytest.mark.asyncio
async def test_runInModalHandlesAsyncGeneratorLsAndStreamsLogs(monkeypatch, tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("print('train')", encoding="utf-8")

    fake_sb = _FakeSandbox()

    async def _lookup(*_args, **_kwargs):
        return object()

    async def _create(*_args, **_kwargs):
        return fake_sb

    fake_modal = SimpleNamespace(
        Image=_FakeImage,
        App=SimpleNamespace(
            lookup=SimpleNamespace(aio=_lookup)
        ),
        Sandbox=SimpleNamespace(
            create=SimpleNamespace(aio=_create)
        ),
        Volume=SimpleNamespace(
            from_name=lambda *_args, **_kwargs: object()
        ),
    )

    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(
        modal_runner,
        "build_image_spec",
        lambda _repo_path, _extra_deps: {"pip_packages": ["numpy"]},
    )

    logs: list[str] = []
    artifact_dir = await modal_runner.run_in_modal(
        repo_path=repo,
        script_content="print('hello')",
        timeout_seconds=30,
        stream_logs=True,
        log_callback=logs.append,
    )

    assert (artifact_dir / "profile_result.json").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert any("Resolving Modal app handle" in msg for msg in logs)
    assert any("Modal run complete" in msg for msg in logs)
    assert any(msg.startswith("[sandbox] hello from sandbox") for msg in logs)

    captured = capsys.readouterr()
    assert "hello from sandbox" in captured.out
