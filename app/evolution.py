"""Community events between consecutive archive snapshots.

Events follow Unit 3: birth, growth, merge, split, shrink, dissolve.
A later community is related to an earlier one when they share members,
using the inclusion rules common in temporal community tracking.
"""

from __future__ import annotations

import networkx as nx


def _related(previous: set[str], current: set[str]) -> bool:
    overlap = len(previous & current)
    if overlap == 0 or not previous or not current:
        return False
    jaccard = overlap / len(previous | current)
    kept = overlap / len(previous)
    explained = overlap / len(current)
    return jaccard >= 0.3 or kept >= 0.5 or explained >= 0.5


def detect_events(previous: list[set[str]], current: list[set[str]]) -> list[dict]:
    previous = [comm for comm in previous if len(comm) >= 2]
    current = [comm for comm in current if len(comm) >= 2]
    if not previous:
        return [_event("birth", sorted(comm), "A community appears in this snapshot.") for comm in current]

    parents_of = [[i for i, prev in enumerate(previous) if _related(prev, comm)] for comm in current]
    children_of: list[list[int]] = [[] for _ in previous]
    for c_index, parents in enumerate(parents_of):
        for p_index in parents:
            children_of[p_index].append(c_index)

    events: list[dict] = []
    claimed_current: set[int] = set()

    for p_index, children in enumerate(children_of):
        if len(children) < 2:
            continue
        events.append(
            _event(
                "split",
                sorted(previous[p_index]),
                "One community divides into sub-communities.",
                sources=[sorted(current[c]) for c in children],
            )
        )
        claimed_current.update(children)

    for c_index, comm in enumerate(current):
        if c_index in claimed_current:
            continue
        parents = parents_of[c_index]
        if len(parents) >= 2:
            events.append(
                _event(
                    "merge",
                    sorted(comm),
                    "Two or more communities combine.",
                    sources=[sorted(previous[p]) for p in parents],
                )
            )
            continue
        if len(parents) == 1:
            prev = previous[parents[0]]
            if len(comm) > len(prev):
                events.append(_event("growth", sorted(comm), f"Membership grew from {len(prev)} to {len(comm)}."))
            elif len(comm) < len(prev):
                events.append(_event("shrink", sorted(comm), f"Membership fell from {len(prev)} to {len(comm)}."))
            continue
        events.append(_event("birth", sorted(comm), "A community appears with little overlap to the previous snapshot."))

    claimed_prev = {p for parents in parents_of for p in parents}
    for p_index, prev in enumerate(previous):
        if p_index not in claimed_prev:
            events.append(_event("dissolve", sorted(prev), "The community disappears from the later snapshot."))

    if not events:
        events.append(_event("stable", [], "This deposit preserves the same community structure."))
    return events


def _event(kind: str, members: list[str], detail: str, sources: list[list[str]] | None = None) -> dict:
    return {"type": kind, "members": members, "detail": detail, "sources": sources or []}


def partition(graph: nx.Graph) -> list[set[str]]:
    if graph.number_of_nodes() == 0:
        return []
    communities = nx.community.louvain_communities(graph, weight="weight", resolution=1.0, seed=7)
    return [set(comm) for comm in communities]


def layout(graph: nx.Graph) -> dict[str, dict[str, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    pos = nx.spring_layout(graph, seed=7, weight="weight", k=1.6)
    xs = [point[0] for point in pos.values()]
    ys = [point[1] for point in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale(value: float, low: float, high: float) -> float:
        if high - low < 1e-6:
            return 0.5
        return 0.14 + 0.72 * (value - low) / (high - low)

    return {
        node: {"x": round(float(scale(point[0], min_x, max_x)), 4), "y": round(float(scale(point[1], min_y, max_y)), 4)}
        for node, point in pos.items()
    }


def graph_stats(graph: nx.Graph, communities: list[set[str]]) -> dict:
    groups = [comm for comm in communities if len(comm) >= 2] or communities
    modularity = 0.0
    if graph.number_of_edges() and len(communities) > 1:
        modularity = float(nx.community.modularity(graph, communities, weight="weight"))
    degrees = [degree for _, degree in graph.degree()]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "communities": len(groups),
        "modularity": round(modularity, 3),
        "mean_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0.0,
    }
