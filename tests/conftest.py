import pytest

from launcher import app as launcher_app


@pytest.fixture(autouse=True)
def _no_git_backoff_or_network(monkeypatch):
    """Git retries sleep 5/15/30s and fall back to a GitHub download; neither in tests."""
    monkeypatch.setattr(launcher_app, "GIT_RETRY_DELAYS", (0.0, 0.0, 0.0))

    async def no_download(repo, ref, destination):
        raise RuntimeError("archive download disabled in tests")

    monkeypatch.setattr(launcher_app, "download_github_tarball", no_download)
