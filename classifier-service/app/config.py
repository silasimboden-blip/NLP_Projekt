import os

DEFAULT_LABELS = [
    "Zustimmung",
    "sachliche Kritik",
    "Empörung",
    "Beleidigung oder persönlicher Angriff",
]

MODEL_ID = os.getenv("MODEL_ID", "facebook/bart-large-mnli")

_labels_raw = os.getenv("CANDIDATE_LABELS")
CANDIDATE_LABELS = (
    [s.strip() for s in _labels_raw.split(",") if s.strip()]
    if _labels_raw
    else DEFAULT_LABELS
)

BATCH_WINDOW_MS = int(os.getenv("BATCH_WINDOW_MS", "200"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "256"))
