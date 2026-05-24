import os
import importlib


def reload_config():
    """Re-import config.py so it re-reads env vars."""
    import app.config
    return importlib.reload(app.config)


def test_defaults(monkeypatch):
    for var in [
        "MODEL_ID", "CANDIDATE_LABELS", "BATCH_WINDOW_MS",
        "MAX_BATCH_SIZE", "MAX_QUEUE_SIZE",
    ]:
        monkeypatch.delenv(var, raising=False)
    cfg = reload_config()
    assert cfg.MODEL_ID == "facebook/bart-large-mnli"
    assert cfg.CANDIDATE_LABELS == [
        "Zustimmung",
        "sachliche Kritik",
        "Empörung",
        "Beleidigung oder persönlicher Angriff",
    ]
    assert cfg.BATCH_WINDOW_MS == 200
    assert cfg.MAX_BATCH_SIZE == 8
    assert cfg.MAX_QUEUE_SIZE == 256


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CANDIDATE_LABELS", "a,b,c")
    monkeypatch.setenv("BATCH_WINDOW_MS", "50")
    monkeypatch.setenv("MAX_BATCH_SIZE", "4")
    cfg = reload_config()
    assert cfg.CANDIDATE_LABELS == ["a", "b", "c"]
    assert cfg.BATCH_WINDOW_MS == 50
    assert cfg.MAX_BATCH_SIZE == 4
