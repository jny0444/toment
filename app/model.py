"""PyTorch multi-label toxicity head on hashed n-grams plus lexicon slots.

Required stack from the brief: PyTorch, NumPy, Pandas. Scores are probabilities
for the six Jigsaw labels. A label is flagged at 0.70. Official ranking
metric is ROC-AUC, which does not use that cutoff.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from app.corpus import ENGLISH_HOLDOUT, MULTILINGUAL_CHALLENGE, build_training_frame, holdout_frame
from app.lexicon import LABELS, lexicon_hits

BUCKETS = 2048
FEATURE_DIM = BUCKETS + len(LABELS)
# Unseen languages sit near 0.6 when no cue fires. True toxic rows in the
# hold-out sets score at least 0.80, so 0.70 keeps those and drops the rest.
THRESHOLD = 0.70
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "toxicity.pt"
TOKEN = re.compile(r"\w+", re.UNICODE)


class ToxicityNet(nn.Module):
    def __init__(self, d_in: int = FEATURE_DIM, d_hidden: int = 128, d_out: int = len(LABELS)):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _bucket(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % BUCKETS


def featurize(text: str) -> np.ndarray:
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    folded = f" {text.casefold()} "
    for i in range(len(folded) - 2):
        vec[_bucket(folded[i : i + 3])] += 1.0
    words = TOKEN.findall(text.casefold())
    for word in words:
        vec[_bucket(f"w:{word}")] += 1.5
    for left, right in zip(words, words[1:]):
        vec[_bucket(f"b:{left}_{right}")] += 1.5
    vec[:BUCKETS] = np.log1p(vec[:BUCKETS])
    norm = np.linalg.norm(vec[:BUCKETS])
    if norm > 0:
        vec[:BUCKETS] /= norm
    hits = lexicon_hits(text)
    for i, label in enumerate(LABELS):
        vec[BUCKETS + i] = np.log1p(hits[label])
    return vec


def _matrix(texts: list[str]) -> np.ndarray:
    return np.stack([featurize(text) for text in texts])


class ToxicityModel:
    def __init__(self) -> None:
        self.net = ToxicityNet()
        self.metrics: dict = {}
        self.train_frame: pd.DataFrame | None = None

    def fit(self, epochs: int = 40, seed: int = 7) -> dict:
        torch.manual_seed(seed)
        frame = build_training_frame(seed=seed)
        self.train_frame = frame
        x = torch.tensor(_matrix(frame["text"].tolist()), dtype=torch.float32)
        y = torch.tensor(frame[list(LABELS)].to_numpy(dtype=np.float32), dtype=torch.float32)
        positive = y.sum(dim=0)
        weight = (y.shape[0] - positive) / positive.clamp(min=1.0)

        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(x), y)
            loss.backward()
            opt.step()

        self.net.eval()
        self.metrics = self.evaluate()
        self.metrics["train_rows"] = int(len(frame))
        self.metrics["train_loss"] = float(loss.detach())
        prevalence = frame[list(LABELS)].mean().round(3).to_dict()
        self.metrics["label_prevalence"] = {key: float(value) for key, value in prevalence.items()}
        return self.metrics

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "metrics": self.metrics}, path)

    def load(self, path: Path = MODEL_PATH) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(blob["state"])
        self.net.eval()
        self.metrics = blob.get("metrics", {})

    @torch.no_grad()
    def predict_proba(self, texts: list[str], use_lexicon: bool = True) -> np.ndarray:
        self.net.eval()
        if not texts:
            return np.zeros((0, len(LABELS)), dtype=np.float32)
        x = torch.tensor(_matrix(texts), dtype=torch.float32)
        if not use_lexicon:
            x[:, BUCKETS:] = 0
        return torch.sigmoid(self.net(x)).numpy()

    def classify(self, text: str) -> dict:
        probs = self.predict_proba([text])[0]
        ngram = self.predict_proba([text], use_lexicon=False)[0]
        scores = {label: round(float(probs[i]), 4) for i, label in enumerate(LABELS)}
        ngram_scores = {label: round(float(ngram[i]), 4) for i, label in enumerate(LABELS)}
        hits = lexicon_hits(text)
        flagged = [label for label, score in scores.items() if score >= THRESHOLD]
        return {
            "text": text,
            "scores": scores,
            "ngram_scores": ngram_scores,
            "flagged": flagged,
            "toxic": bool(flagged),
            "signals": {
                "ngram": bool(ngram_scores["toxic"] >= THRESHOLD),
                "lexicon": any(count > 0 for count in hits.values()),
                "lexicon_hits": hits,
            },
        }

    def _auc_pack(self, rows: list[dict], name: str, use_lexicon: bool) -> dict:
        frame = holdout_frame(rows)
        y_true = frame[list(LABELS)].to_numpy(dtype=np.float32)
        y_score = self.predict_proba(frame["text"].tolist(), use_lexicon=use_lexicon)
        per_label = {}
        for i, label in enumerate(LABELS):
            if len(np.unique(y_true[:, i])) < 2:
                continue
            per_label[label] = round(float(roc_auc_score(y_true[:, i], y_score[:, i])), 3)
        macro = round(float(np.mean(list(per_label.values()))), 3) if per_label else None
        return {"name": name, "rows": int(len(frame)), "macro_auc": macro, "per_label_auc": per_label}

    def evaluate(self) -> dict:
        return {
            "english_holdout": self._auc_pack(ENGLISH_HOLDOUT, "english_holdout", True),
            "multilingual_challenge": self._auc_pack(MULTILINGUAL_CHALLENGE, "multilingual_challenge", True),
            "multilingual_ngram_only": self._auc_pack(MULTILINGUAL_CHALLENGE, "multilingual_ngram_only", False),
        }


def prepare_model() -> ToxicityModel:
    model = ToxicityModel()
    if MODEL_PATH.exists():
        model.load()
        if "english_holdout" not in model.metrics:
            model.metrics = model.evaluate()
        return model
    model.fit()
    model.save()
    return model
