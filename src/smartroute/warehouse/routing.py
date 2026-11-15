"""
Model Training - routing policies (baselines) and Evaluation - exact optimal benchmark.

Baseline routing heuristics (Hall, 1993; Petersen, 1997; Roodbergen and de Koster, 2001),
adapted to the two-block layout in which the central cross aisle is the front of both blocks
and contains the depot:

* S-shape    : every aisle containing picks is traversed completely; if the number of pick
               aisles in a block is odd, the last aisle is entered and left from the front.
* Return     : every aisle containing picks is entered and left from the front cross aisle,
               going only as deep as the deepest pick.
* LargestGap : the first and last pick aisles of a block are traversed completely; every
               aisle in between is entered from the front and from the back up to the largest
               gap between adjacent picks (or between a pick and an aisle end).

Blocks are served sequentially (left block, then right block). Within a block the aisles
are swept from the extreme aisle nearest to the picker's current position.

Optimal benchmark: the exact travelling-salesman tour over the depot and all pick points on
the shortest-path aisle network. Small instances are solved with Held-Karp dynamic
programming; larger ones with an integer program (Dantzig, Fulkerson and Johnson, 1954)
solved by HiGHS with iterative subtour-elimination cuts. The optimum is a benchmark (upper
bound on route quality), not a training label.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

from smartroute import config
from smartroute.warehouse.layout import path_length


@dataclass
class RouteResult:
    """Output of a routing policy for one pick list."""
    policy: str
    waypoints: list
    distance_units: float
    compute_time_s: float
    is_optimal_proven: bool = True
    visit_order: list = field(default_factory=list)

    @property
    def distance_m(self):
        return self.distance_units * config.UNIT_TO_METRES


# =============================================================================
# MODEL TRAINING
# =============================================================================

# --- Baseline routing heuristics ------------------------------------------------
def _group_by_aisle(layout, points):
    """{block: {aisle_y: sorted unique depths}} for the pick points of one wave."""
    grouped = defaultdict(lambda: defaultdict(set))
    for p in points:
        grouped[p.block][p.aisle_y].add(round(layout.depth(p), 6))
    return {b: {y: sorted(d) for y, d in aisles.items()} for b, aisles in grouped.items()}


def _x_at_depth(layout, block, depth):
    return layout.front_x - depth if block == "L" else layout.front_x + depth


def _ordered_aisles(aisle_ys, current_y):
    """Sweep aisles from the extreme nearest to the current position."""
    ys = sorted(aisle_ys)
    if abs(current_y - ys[0]) <= abs(current_y - ys[-1]):
        return ys
    return ys[::-1]


def _largest_gap_reaches(depths, aisle_length):
    """Return (front_reach, back_reach) depths for one intermediate aisle."""
    positions = [0.0] + list(depths) + [aisle_length]
    gaps = np.diff(positions)
    j = int(np.argmax(gaps))                      # gap between positions[j] and positions[j+1]
    front_reach = positions[j] if j > 0 else None
    back_reach = positions[j + 1] if j + 1 < len(positions) - 1 else None
    return front_reach, back_reach


def _block_path(layout, policy, block, aisles, current_y):
    """Waypoints that serve one block, starting and ending on the front cross aisle."""
    fx = layout.front_x
    bx = layout.back_x[block]
    length = layout.aisle_length[block]
    order = _ordered_aisles(aisles.keys(), current_y)
    path = []

    if policy == "Return" or (policy == "LargestGap" and len(order) == 1):
        for y in order:
            deepest = _x_at_depth(layout, block, aisles[y][-1])
            path += [(fx, y), (deepest, y), (fx, y)]
        return path

    if policy == "SShape":
        at_front = True
        for i, y in enumerate(order):
            is_last = i == len(order) - 1
            if at_front and is_last:
                deepest = _x_at_depth(layout, block, aisles[y][-1])
                path += [(fx, y), (deepest, y), (fx, y)]
            elif at_front:
                path += [(fx, y), (bx, y)]
                at_front = False
            else:
                path += [(bx, y), (fx, y)]
                at_front = True
        return path

    if policy == "LargestGap":
        first, last, middle = order[0], order[-1], order[1:-1]
        reaches = {y: _largest_gap_reaches(aisles[y], length) for y in middle}
        path += [(fx, first), (bx, first)]                     # traverse the first aisle
        for y in middle:                                        # back-side visits
            back_reach = reaches[y][1]
            if back_reach is not None:
                path += [(bx, y), (_x_at_depth(layout, block, back_reach), y), (bx, y)]
        path += [(bx, last), (fx, last)]                        # traverse the last aisle
        for y in reversed(middle):                              # front-side visits
            front_reach = reaches[y][0]
            if front_reach is not None:
                path += [(fx, y), (_x_at_depth(layout, block, front_reach), y), (fx, y)]
        return path

    raise ValueError(f"Unknown routing policy: {policy}")


def route_heuristic(layout, points, policy):
    """Model Training - build the route of a baseline heuristic for one pick list."""
    start = time.perf_counter()
    depot = layout.depot.xy
    grouped = _group_by_aisle(layout, points)
    waypoints = [depot]
    current_y = depot[1]
    for block in ("L", "R"):
        if block not in grouped:
            continue
        block_path = _block_path(layout, policy, block, grouped[block], current_y)
        waypoints += [(layout.front_x, current_y)] + block_path
        current_y = waypoints[-1][1]
    waypoints += [(layout.front_x, current_y), depot]
    waypoints = _dedupe(waypoints)
    distance = path_length(waypoints)
    return RouteResult(policy, waypoints, distance, time.perf_counter() - start)


def _dedupe(waypoints):
    cleaned = [waypoints[0]]
    for wp in waypoints[1:]:
        if wp != cleaned[-1]:
            cleaned.append(wp)
    return cleaned


# =============================================================================
# EVALUATION
# =============================================================================

# --- Optimal benchmark (exact TSP) ---------------------------------------------
def _held_karp(distance_matrix):
    """Exact TSP by dynamic programming; node 0 is the depot."""
    n = len(distance_matrix)
    m = n - 1
    num_masks = 1 << m
    dp = np.full((num_masks, m), np.inf)
    parent = np.full((num_masks, m), -1, dtype=int)
    for j in range(m):
        dp[1 << j, j] = distance_matrix[0, j + 1]
    for mask in range(1, num_masks):
        for j in range(m):
            if not (mask >> j) & 1 or not np.isfinite(dp[mask, j]):
                continue
            base = dp[mask, j]
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                new_mask = mask | (1 << k)
                candidate = base + distance_matrix[j + 1, k + 1]
                if candidate < dp[new_mask, k]:
                    dp[new_mask, k] = candidate
                    parent[new_mask, k] = j
    full = num_masks - 1
    totals = dp[full] + distance_matrix[1:, 0]
    last = int(np.argmin(totals))
    order, mask = [], full
    while last != -1:
        order.append(last + 1)
        previous = parent[mask, last]
        mask ^= 1 << last
        last = previous
    return [0] + order[::-1], float(totals.min())


def _milp_tsp(distance_matrix, time_limit):
    """Exact symmetric TSP: degree constraints + lazy subtour-elimination cuts (HiGHS)."""
    n = len(distance_matrix)
    edges = list(itertools.combinations(range(n), 2))
    costs = np.array([distance_matrix[i, j] for i, j in edges])
    degree = lil_matrix((n, len(edges)))
    for e, (i, j) in enumerate(edges):
        degree[i, e] = 1
        degree[j, e] = 1
    constraints = [LinearConstraint(degree.tocsr(), 2, 2)]
    deadline = time.perf_counter() + time_limit
    proven = True
    while True:
        remaining = max(deadline - time.perf_counter(), 0.5)
        result = milp(costs, constraints=constraints, integrality=np.ones(len(edges)),
                      bounds=Bounds(0, 1), options={"time_limit": remaining, "disp": False})
        if result.x is None:
            raise RuntimeError("MILP failed to return a solution")
        chosen = [edges[e] for e in np.flatnonzero(result.x > 0.5)]
        adjacency = lil_matrix((n, n))
        for i, j in chosen:
            adjacency[i, j] = adjacency[j, i] = 1
        num_components, labels = connected_components(adjacency.tocsr(), directed=False)
        if num_components == 1:
            proven = proven and result.status == 0
            break
        if time.perf_counter() > deadline:
            proven = False
            break
        for c in range(num_components):
            members = set(np.flatnonzero(labels == c))
            row = np.array([1.0 if (i in members and j in members) else 0.0 for i, j in edges])
            constraints.append(LinearConstraint(row.reshape(1, -1), -np.inf, len(members) - 1))
    # rebuild the tour from the chosen edges
    neighbours = defaultdict(list)
    for i, j in chosen:
        neighbours[i].append(j)
        neighbours[j].append(i)
    tour, previous, current = [0], None, 0
    while len(tour) < n:
        nxt = [v for v in neighbours[current] if v != previous][0]
        if nxt == 0:
            break
        tour.append(nxt)
        previous, current = current, nxt
    length = sum(distance_matrix[tour[k], tour[(k + 1) % len(tour)]] for k in range(len(tour)))
    return tour, float(length), proven


def unique_points(points):
    """Collapse pick points that share the same aisle position (different levels/slots)."""
    seen, unique = set(), []
    for p in points:
        key = (p.block, p.aisle_y, round(p.x, 6))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def route_optimal(layout, points, time_limit=config.OPTIMAL_TIME_LIMIT_S, with_waypoints=True):
    """Evaluation - exact optimal tour over the depot and every pick point."""
    start = time.perf_counter()
    nodes = [layout.depot] + unique_points(points)
    matrix = layout.distance_matrix(nodes)
    proven = True
    if len(nodes) == 1:
        tour, length = [0], 0.0
    elif len(nodes) <= 3:
        tour, length = list(range(len(nodes))), float(
            sum(matrix[k, (k + 1) % len(nodes)] for k in range(len(nodes))))
    elif len(nodes) <= 10:
        tour, length = _held_karp(matrix)
    else:
        tour, length, proven = _milp_tsp(matrix, time_limit)
    elapsed = time.perf_counter() - start
    waypoints = []
    if with_waypoints:
        sequence = [nodes[k] for k in tour] + [nodes[0]]
        waypoints = [sequence[0].xy]
        for a, b in zip(sequence[:-1], sequence[1:]):
            waypoints += layout.shortest_path_waypoints(a, b)[1:]
        waypoints = _dedupe(waypoints)
    return RouteResult("Optimal", waypoints, length, elapsed, proven, visit_order=tour)


def route_all(layout, points, include_optimal=True):
    """Evaluation - run every baseline (and optionally the optimum) for one pick list."""
    results = {policy: route_heuristic(layout, points, policy) for policy in config.ROUTING_POLICIES}
    if include_optimal:
        results["Optimal"] = route_optimal(layout, points)
    return results
