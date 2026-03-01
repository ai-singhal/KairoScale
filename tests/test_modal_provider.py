"""Tests for ModalProvider auto-deploy behavior."""

from KairoScale.agent.providers import modal_provider
from KairoScale.agent.providers.modal_provider import ModalProvider


def test_modalProviderUsesExistingEnvUrl(monkeypatch):
    monkeypatch.setenv("MODAL_VLLM_URL", "https://example.modal.run")

    def _fail_deploy(_model: str) -> str:
        raise AssertionError("deploy_modal_vllm should not be called when URL exists")

    monkeypatch.setattr(modal_provider, "deploy_modal_vllm", _fail_deploy)

    provider = ModalProvider(model="Qwen/Qwen3-8B")
    assert provider.base_url == "https://example.modal.run"


def test_modalProviderAutoDeploysWhenUrlMissing(monkeypatch):
    monkeypatch.delenv("MODAL_VLLM_URL", raising=False)
    monkeypatch.setattr(
        modal_provider,
        "deploy_modal_vllm",
        lambda _model: "https://autodeployed.modal.run",
    )

    provider = ModalProvider(model="Qwen/Qwen3-8B")

    assert provider.base_url == "https://autodeployed.modal.run"
    assert modal_provider.os.environ.get("MODAL_VLLM_URL") == "https://autodeployed.modal.run"
