from typing import Callable, List, Tuple


class Classifier:
    """Thin wrapper around an HF zero-shot pipeline.

    Constructor takes the pipeline callable so tests can inject a fake
    without downloading the HF model.
    """

    def __init__(self, pipeline: Callable, candidate_labels: List[str]):
        self._pipeline = pipeline
        self._labels = candidate_labels

    def classify_batch(self, comments: List[str]) -> List[Tuple[str, float]]:
        if not comments:
            return []
        outputs = self._pipeline(
            comments,
            candidate_labels=self._labels,
            multi_label=False,
        )
        return [(o["labels"][0], float(o["scores"][0])) for o in outputs]


def build_real_classifier() -> Classifier:
    """Construct a Classifier backed by the real HF pipeline.

    Imported lazily so unit tests don't drag transformers/torch in.
    """
    from transformers import pipeline as hf_pipeline
    from app.config import MODEL_ID, CANDIDATE_LABELS

    hf = hf_pipeline("zero-shot-classification", model=MODEL_ID)
    return Classifier(pipeline=hf, candidate_labels=CANDIDATE_LABELS)
