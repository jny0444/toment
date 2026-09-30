# FOLIO

A working prototype for the multilingual toxic-comment problem, read through the three communities in Unit 3: a political movement, a scientific collaboration, and a health-rumour cascade.

The desk treats each community as a web-archive deposit. Louvain finds communities inside a snapshot. The next deposit is compared with the previous one for birth, growth, merge, split, shrink, and dissolve. Comments under the graph are scored on the six Jigsaw labels: toxic, severe toxic, obscene, threat, insult, and identity hate.

## Stack

- PyTorch multi-label classifier (hashed character and word n-grams, English training only)
- NumPy feature vectors, Pandas training frame and label prevalence
- NetworkX Louvain partitions, modularity, and snapshot layout
- scikit-learn ROC-AUC, the ranking metric from the brief
- FastAPI for the archive desk

The full competition models in the brief are multilingual DistilBERT and XLM-RoBERTa. This prototype stays on CPU: the same six lexicon slots are shared across languages, so a weight learned from English still fires for Spanish, French, German, Hindi, or Portuguese insults. The header reports three AUC figures: English hold-out, multilingual challenge with those slots, and multilingual challenge with n-grams only. The third number is the English-only gap.

## Run

Use Python 3.12. PyTorch does not install cleanly on 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

The first launch trains the model and writes `models/toxicity.pt`. Later launches load that file. Reset deposits in the desk to restore the three collections.

## Try

- Civic mobilisation: Early (two communities), Mid (one bridge, south forum grows), Late (the forums merge).
- Research collaborations: Initial, the archived copy of that graph, then Evolved, where new co-authorships join the labs.
- Health rumour cascade: Early, Growth, then Intervention. Cut toxic paths on the growth deposit to drop edges touching toxic accounts.
- File a comment in another language, for example `Todos ustedes son unos idiotas` or `Ich bringe dich um`.
