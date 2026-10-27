"""Transportation-network loading and shortest-path utilities.

Network files follow the format used by the paper's released dataset
(From, To, Volume, Cost).  ``Cost`` is the mean travel time of a link.
Node identifiers in the files are 1-based and may be non-contiguous;
they are remapped to ``0 .. n_nodes - 1`` internally.
"""

from __future__ import annotations

import csv
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

INF = float("inf")


@dataclass(frozen=True)
class Link:
    """A directed link with its mean travel time."""

    index: int
    u: int          # tail node (internal index)
    v: int          # head node (internal index)
    cost: float     # mean travel time


class Network:
    """Directed, connected-in-the-relevant-sense transportation network."""

    def __init__(
        self,
        name: str,
        node_ids: Sequence[int],
        links: Sequence[Link],
        out_links: Sequence[Sequence[int]],
        in_links: Sequence[Sequence[int]],
    ) -> None:
        self.name = name
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.links = list(links)
        self.out_links = [list(x) for x in out_links]
        self.in_links = [list(x) for x in in_links]
        self.n_nodes = len(node_ids)
        self.n_links = len(links)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None) -> "Network":
        path = Path(path)
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            rows = list(reader)
        if not rows:
            raise ValueError(f"empty network file: {path}")

        raw_nodes: set[int] = set()
        raw_links: list[tuple[int, int, float]] = []
        for row in rows:
            u = int(float(row["From"]))
            v = int(float(row["To"]))
            cost = float(row["Cost"])
            if not np.isfinite(cost) or cost < 0:
                raise ValueError(f"invalid link cost in {path}: {row}")
            raw_nodes.add(u)
            raw_nodes.add(v)
            raw_links.append((u, v, cost))

        node_ids = sorted(raw_nodes)
        remap = {orig: idx for idx, orig in enumerate(node_ids)}
        links: list[Link] = []
        out_links: list[list[int]] = [[] for _ in node_ids]
        in_links: list[list[int]] = [[] for _ in node_ids]
        for idx, (u_raw, v_raw, cost) in enumerate(raw_links):
            u, v = remap[u_raw], remap[v_raw]
            links.append(Link(index=idx, u=u, v=v, cost=cost))
            out_links[u].append(idx)
            in_links[v].append(idx)

        return cls(
            name=name or path.stem,
            node_ids=node_ids,
            links=links,
            out_links=out_links,
            in_links=in_links,
        )

    @classmethod
    def from_edges(
        cls, name: str, edges: Sequence[tuple[int, int, float]]
    ) -> "Network":
        """Build a network from ``(u, v, cost)`` tuples using arbitrary IDs."""
        raw_nodes = {x for u, v, _ in edges for x in (u, v)}
        node_ids = sorted(raw_nodes)
        remap = {orig: idx for idx, orig in enumerate(node_ids)}
        links: list[Link] = []
        out_links: list[list[int]] = [[] for _ in node_ids]
        in_links: list[list[int]] = [[] for _ in node_ids]
        for idx, (u_raw, v_raw, cost) in enumerate(edges):
            u, v = remap[u_raw], remap[v_raw]
            links.append(Link(index=idx, u=u, v=v, cost=float(cost)))
            out_links[u].append(idx)
            in_links[v].append(idx)
        return cls(name, node_ids, links, out_links, in_links)

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------
    @property
    def mean_costs(self) -> np.ndarray:
        return np.asarray([link.cost for link in self.links], dtype=np.float64)

    @property
    def u(self) -> np.ndarray:
        return np.asarray([link.u for link in self.links], dtype=np.int64)

    @property
    def v(self) -> np.ndarray:
        return np.asarray([link.v for link in self.links], dtype=np.int64)

    def original_id(self, node: int) -> int:
        """Map an internal node index back to the ID used in the data file."""
        return int(self.node_ids[node])

    def internal_id(self, original: int) -> int:
        """Map a data-file node ID to the internal index."""
        pos = int(np.searchsorted(self.node_ids, original))
        if pos >= self.n_nodes or int(self.node_ids[pos]) != original:
            raise KeyError(f"node {original} does not exist in {self.name}")
        return pos

    # ------------------------------------------------------------------
    # Dijkstra
    # ------------------------------------------------------------------
    def dijkstra_to(self, destination: int) -> tuple[np.ndarray, np.ndarray]:
        """Shortest paths from *every* node to ``destination``.

        Returns
        -------
        dist : ndarray, shape (n_nodes,)
            Least expected travel time to the destination (``inf`` if
            unreachable).
        next_link : ndarray, shape (n_nodes,)
            Index of the first link on the shortest path to the destination
            (``-1`` at the destination or when unreachable).  This is exactly
            the least-expected-time (LET) routing policy used to initialise
            FMA.
        """
        dist = np.full(self.n_nodes, INF, dtype=np.float64)
        next_link = np.full(self.n_nodes, -1, dtype=np.int64)
        if not (0 <= destination < self.n_nodes):
            raise IndexError(destination)

        dist[destination] = 0.0
        heap: list[tuple[float, int]] = [(0.0, destination)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist[node]:
                continue
            for lid in self.in_links[node]:
                link = self.links[lid]
                cand = d + link.cost
                if cand < dist[link.u] - 1e-12:
                    dist[link.u] = cand
                    next_link[link.u] = lid
                    heapq.heappush(heap, (cand, link.u))
        return dist, next_link

    def dijkstra_from(self, origin: int) -> tuple[np.ndarray, np.ndarray]:
        """Shortest paths from ``origin`` to every node (forward search)."""
        dist = np.full(self.n_nodes, INF, dtype=np.float64)
        prev_link = np.full(self.n_nodes, -1, dtype=np.int64)
        if not (0 <= origin < self.n_nodes):
            raise IndexError(origin)

        dist[origin] = 0.0
        heap: list[tuple[float, int]] = [(0.0, origin)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist[node]:
                continue
            for lid in self.out_links[node]:
                link = self.links[lid]
                cand = d + link.cost
                if cand < dist[link.v] - 1e-12:
                    dist[link.v] = cand
                    prev_link[link.v] = lid
                    heapq.heappush(heap, (cand, link.v))
        return dist, prev_link

    def let_path_links(self, origin: int, next_link: np.ndarray) -> list[int]:
        """Follow the LET policy from ``origin`` and return its link path."""
        path, _, _ = self.follow_policy_links(origin, next_link)
        return path

    def follow_policy_links(
        self, origin: int, policy_links: np.ndarray
    ) -> tuple[list[int], list[int], bool]:
        """Follow a deterministic link policy until the destination sentinel.

        A policy is an array mapping each node to the index of its next link
        (``-1`` at the destination).  Returns ``(nodes, links, ok)``; ``ok`` is
        False when the walk cycles or reaches an undefined/unreachable link.
        """
        nodes = [origin]
        links: list[int] = []
        seen = {origin}
        current = origin
        while True:
            lid = int(policy_links[current])
            if lid == -1:
                return nodes, links, True
            if lid < 0 or lid >= self.n_links:
                return nodes, links, False
            link = self.links[lid]
            if link.u != current:
                return nodes, links, False
            current = link.v
            links.append(lid)
            if current in seen:
                return nodes, links, False
            seen.add(current)
            nodes.append(current)

    def hops_to_destination(self, origin: int, next_link: np.ndarray) -> int:
        """Number of links on the LET path (``-1`` if unreachable)."""
        nodes, links, ok = self.follow_policy_links(origin, next_link)
        return len(links) if ok else -1

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def node_path(self, link_path: Iterable[int]) -> list[int]:
        links = list(link_path)
        if not links:
            return []
        nodes = [self.links[links[0]].u]
        for lid in links:
            nodes.append(self.links[lid].v)
        return nodes

    def link_between(self, u: int, v: int) -> int:
        """Return the cheapest link from ``u`` to ``v`` (``-1`` if none)."""
        best, best_cost = -1, INF
        for lid in self.out_links[u]:
            if self.links[lid].v == v and self.links[lid].cost < best_cost:
                best, best_cost = lid, self.links[lid].cost
        return best

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Network({self.name!r}, nodes={self.n_nodes}, "
            f"links={self.n_links})"
        )
