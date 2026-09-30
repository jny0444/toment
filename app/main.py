"""FOLIO archive desk: community evolution plus multilingual toxicity."""

from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path

import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.communities import COLLECTIONS
from app.evolution import detect_events, graph_stats, layout, partition
from app.lexicon import LABELS
from app.model import ToxicityModel, prepare_model

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

model: ToxicityModel | None = None
archive: list[dict] = []


def _blank_archive() -> list[dict]:
    collections = deepcopy(COLLECTIONS)
    for collection in collections:
        for snapshot in collection["slices"]:
            snapshot["edges"] = [
                {"source": source, "target": target, "weight": weight, "kind": kind, "cut": False}
                for source, target, weight, kind in snapshot["edges"]
            ]
            for index, comment in enumerate(snapshot["comments"]):
                comment["id"] = f"{snapshot['id']}-{index}"
    return collections


def _analysis_graph(snapshot: dict) -> nx.Graph:
    graph = nx.Graph()
    for node in snapshot["nodes"]:
        graph.add_node(node["id"])
    for edge in snapshot["edges"]:
        if edge["cut"]:
            continue
        graph.add_edge(edge["source"], edge["target"], weight=edge["weight"], kind=edge["kind"])
    return graph


def _layout_graph(snapshot: dict) -> nx.Graph:
    graph = nx.Graph()
    for node in snapshot["nodes"]:
        graph.add_node(node["id"])
    for edge in snapshot["edges"]:
        graph.add_edge(edge["source"], edge["target"], weight=max(edge["weight"], 0.2))
    return graph


def _decorate_comments(snapshot: dict) -> list[dict]:
    assert model is not None
    decorated = []
    for comment in snapshot["comments"]:
        scored = model.classify(comment["text"])
        decorated.append({**comment, **{key: scored[key] for key in ("scores", "ngram_scores", "flagged", "toxic", "signals")}})
    return decorated


def _view_slice(snapshot: dict, previous_communities: list[set[str]] | None) -> dict:
    comments = _decorate_comments(snapshot)
    analysis = _analysis_graph(snapshot)
    communities = partition(analysis)
    positions = layout(_layout_graph(snapshot))
    stats = graph_stats(analysis, communities)
    toxic_authors = {comment["author"] for comment in comments if comment["toxic"]}
    stats["toxic_comments"] = sum(1 for comment in comments if comment["toxic"])
    stats["comments"] = len(comments)
    stats["toxic_share"] = round(stats["toxic_comments"] / stats["comments"], 3) if comments else 0.0

    ordered = sorted(communities, key=lambda comm: (-len(comm), min(comm)))
    color_of = {}
    group_index = 0
    for comm in ordered:
        if len(comm) < 2:
            for node in comm:
                color_of[node] = -1
            continue
        for node in comm:
            color_of[node] = group_index
        group_index += 1

    incoming = detect_events(previous_communities or [], [set(comm) for comm in ordered])
    nodes = []
    for node in snapshot["nodes"]:
        nodes.append(
            {
                **node,
                **positions.get(node["id"], {"x": 0.5, "y": 0.5}),
                "community": color_of.get(node["id"], -1),
                "toxic": node["id"] in toxic_authors,
            }
        )
    return {
        "id": snapshot["id"],
        "label": snapshot["label"],
        "when": snapshot["when"],
        "note": snapshot["note"],
        "nodes": nodes,
        "edges": snapshot["edges"],
        "comments": comments,
        "stats": stats,
        "events": incoming,
        "communities": [sorted(comm) for comm in ordered if len(comm) >= 2],
        "_communities": [set(comm) for comm in ordered],
    }


def view_collection(collection: dict) -> dict:
    views = []
    previous: list[set[str]] | None = None
    for snapshot in collection["slices"]:
        view = _view_slice(snapshot, previous)
        previous = view["_communities"]
        views.append(view)
    for view in views:
        view.pop("_communities", None)
    return {
        "id": collection["id"],
        "accession": collection["accession"],
        "title": collection["title"],
        "kicker": collection["kicker"],
        "summary": collection["summary"],
        "palette": collection["palette"],
        "slices": views,
    }


def _find(collection_id: str, slice_id: str) -> tuple[dict, dict]:
    for collection in archive:
        if collection["id"] == collection_id:
            for snapshot in collection["slices"]:
                if snapshot["id"] == slice_id:
                    return collection, snapshot
            raise HTTPException(status_code=404, detail="Snapshot not found")
    raise HTTPException(status_code=404, detail="Collection not found")


def guess_lang(text: str) -> str:
    if any("\u0900" <= char <= "\u097f" for char in text):
        return "hi"
    folded = text.casefold()
    probes = {
        "es": ("gracias", "esto", "todos", "cállate", "callate", "mierda"),
        "fr": ("vous", "merci", "graphe", "gueule", "dans"),
        "de": ("die", "und", "nicht", "fresse", "zurück", "zuruck"),
        "pt": ("arquivo", "para", "você", "voce", "cala"),
    }
    for code, words in probes.items():
        if any(word in folded for word in words):
            return code
    return "en"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, archive
    model = prepare_model()
    archive = _blank_archive()
    yield


app = FastAPI(title="FOLIO", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/archive")
def get_archive() -> dict:
    assert model is not None
    return {"collections": [view_collection(collection) for collection in archive], "metrics": model.metrics, "labels": list(LABELS)}


@app.get("/api/metrics")
def get_metrics() -> dict:
    assert model is not None
    return model.metrics


@app.post("/api/reset")
def reset() -> dict:
    global archive
    archive = _blank_archive()
    return get_archive()


@app.post("/api/classify")
def classify(body: dict) -> dict:
    assert model is not None
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    scored = model.classify(text)
    scored["lang"] = guess_lang(text)
    return scored


@app.post("/api/collections/{collection_id}/slices/{slice_id}/comments")
def add_comment(collection_id: str, slice_id: str, body: dict) -> dict:
    _collection, snapshot = _find(collection_id, slice_id)
    text = str(body.get("text", "")).strip()
    author = str(body.get("author", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    known = {node["id"] for node in snapshot["nodes"]}
    if author not in known:
        raise HTTPException(status_code=400, detail="Author is not in this snapshot")
    snapshot["comments"].append(
        {
            "id": f"{slice_id}-u{len(snapshot['comments'])}",
            "author": author,
            "lang": guess_lang(text),
            "text": text,
        }
    )
    return view_collection(_collection)


@app.post("/api/collections/{collection_id}/slices/{slice_id}/moderate")
def moderate(collection_id: str, slice_id: str) -> dict:
    assert model is not None
    collection, snapshot = _find(collection_id, slice_id)
    toxic_authors = set()
    for comment in snapshot["comments"]:
        if model.classify(comment["text"])["toxic"]:
            toxic_authors.add(comment["author"])
    for edge in snapshot["edges"]:
        if edge["source"] in toxic_authors or edge["target"] in toxic_authors:
            edge["cut"] = True
    return view_collection(collection)
