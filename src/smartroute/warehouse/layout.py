"""
Preprocessing and Feature Engineering - warehouse geometry (domain model).

The footwear warehouse is modelled as a two-block, picker-to-parts layout:

* Picking aisles run horizontally (along x). Their centre lines are the y-values of the
  navigation support points (LC/CC/RC labels in dataset_Support_Points_Navigation.csv).
* Three vertical cross aisles connect the aisles: the left cross aisle (x = 66), the central
  cross aisle (x = 403) and the right cross aisle (x = 686).
* The central cross aisle is the *front* of both blocks and contains the I/O depot (CC-08).
  The left block spans x in [66, 403] for all 17 aisles; the right block spans x in [403, 686]
  for the 10 aisles with y >= 631 (an L-shaped building).

A storage location is projected onto its aisle centre line, giving a pick point
(block, aisle_y, x). Travel between pick points is the shortest path in the aisle network,
computed once with Floyd-Warshall over the aisle-end nodes.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

from smartroute import config


@dataclass(frozen=True)
class PickPoint:
    """A picking stop projected onto an aisle centre line."""
    block: str          # "L" (left block) or "R" (right block)
    aisle_y: float      # y of the aisle centre line
    x: float            # position along the aisle

    @property
    def xy(self):
        return (self.x, self.aisle_y)


@dataclass
class WarehouseLayout:
    """Two-block warehouse network with shortest-path distances (layout units)."""
    aisle_ys: list
    right_aisle_ys: list
    front_x: float = config.FRONT_CROSS_AISLE_X
    back_x: dict = field(default_factory=lambda: {"L": config.LEFT_BACK_CROSS_AISLE_X,
                                                    "R": config.RIGHT_BACK_CROSS_AISLE_X})
    depot_xy: tuple = config.DEPOT_XY

    def __post_init__(self):
        self.aisle_ys = sorted(float(y) for y in self.aisle_ys)
        self.right_aisle_ys = sorted(float(y) for y in self.right_aisle_ys)
        self.aisle_length = {b: abs(self.front_x - x) for b, x in self.back_x.items()}
        self._build_node_graph()
        self.depot = PickPoint("L", float(self.depot_xy[1]), float(self.depot_xy[0]))

    # --- network ----------------------------------------------------------------
    def _build_node_graph(self):
        """Nodes are aisle ends: ('F', y) on the central aisle, ('BL', y) and ('BR', y)."""
        nodes = [("F", y) for y in self.aisle_ys]
        nodes += [("BL", y) for y in self.aisle_ys]
        nodes += [("BR", y) for y in self.right_aisle_ys]
        self.node_index = {node: i for i, node in enumerate(nodes)}
        self.nodes = nodes
        graph = lil_matrix((len(nodes), len(nodes)))

        def connect(a, b, length):
            i, j = self.node_index[a], self.node_index[b]
            graph[i, j] = graph[j, i] = max(length, 1e-9)

        for kind, ys in (("F", self.aisle_ys), ("BL", self.aisle_ys), ("BR", self.right_aisle_ys)):
            for y1, y2 in zip(ys[:-1], ys[1:]):
                connect((kind, y1), (kind, y2), y2 - y1)           # along a cross aisle
        for y in self.aisle_ys:
            connect(("F", y), ("BL", y), self.aisle_length["L"])   # through a left aisle
        for y in self.right_aisle_ys:
            connect(("F", y), ("BR", y), self.aisle_length["R"])   # through a right aisle
        self.node_distances, self.node_predecessors = shortest_path(
            graph.tocsr(), method="FW", directed=False, return_predecessors=True)

    def end_nodes(self, point):
        """The two aisle-end nodes of the aisle segment holding `point`."""
        back = "BL" if point.block == "L" else "BR"
        return [(("F", point.aisle_y), self.front_x),
                ((back, point.aisle_y), self.back_x[point.block])]

    def distance(self, p, q):
        """Shortest network distance between two pick points (layout units)."""
        if p.block == q.block and p.aisle_y == q.aisle_y:
            return abs(p.x - q.x)
        best = np.inf
        for node_p, x_p in self.end_nodes(p):
            for node_q, x_q in self.end_nodes(q):
                d = (abs(p.x - x_p) + self.node_distances[self.node_index[node_p], self.node_index[node_q]]
                     + abs(q.x - x_q))
                best = min(best, d)
        return best

    def distance_matrix(self, points):
        """Symmetric distance matrix for a list of pick points."""
        num_points = len(points)
        matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(i + 1, num_points):
                matrix[i, j] = matrix[j, i] = self.distance(points[i], points[j])
        return matrix

    def shortest_path_waypoints(self, p, q):
        """Corridor waypoints (x, y) of the shortest path from p to q (for drawing routes)."""
        if p.block == q.block and p.aisle_y == q.aisle_y:
            return [p.xy, q.xy]
        best, best_pair = np.inf, None
        for node_p, x_p in self.end_nodes(p):
            for node_q, x_q in self.end_nodes(q):
                d = (abs(p.x - x_p) + self.node_distances[self.node_index[node_p], self.node_index[node_q]]
                     + abs(q.x - x_q))
                if d < best:
                    best, best_pair = d, (node_p, node_q)
        i, j = self.node_index[best_pair[0]], self.node_index[best_pair[1]]
        chain = [j]
        while chain[-1] != i:
            chain.append(self.node_predecessors[i, chain[-1]])
        chain.reverse()
        waypoints = [p.xy] + [self.node_xy(self.nodes[k]) for k in chain] + [q.xy]
        return waypoints

    def node_xy(self, node):
        kind, y = node
        x = {"F": self.front_x, "BL": self.back_x["L"], "BR": self.back_x["R"]}[kind]
        return (x, y)

    # --- projection ---------------------------------------------------------------
    def to_pick_point(self, x, y):
        """Project a storage location (x, y) onto its aisle centre line."""
        block = "L" if x <= self.front_x else "R"
        candidates = self.aisle_ys if block == "L" else self.right_aisle_ys
        aisle_y = min(candidates, key=lambda c: abs(c - y))
        return PickPoint(block, aisle_y, float(x))

    def depth(self, point):
        """Distance from the front (central) cross aisle into the aisle."""
        return abs(point.x - self.front_x)


def build_layout(support_points_df):
    """Preprocessing - build the WarehouseLayout from the navigation support points."""
    left_ys = support_points_df.loc[support_points_df["label"].str.startswith("LC"), "y"]
    right_ys = support_points_df.loc[support_points_df["label"].str.startswith("RC"), "y"]
    return WarehouseLayout(aisle_ys=left_ys.tolist(), right_aisle_ys=right_ys.tolist())


def path_length(waypoints):
    """Length of a corridor path given as (x, y) waypoints (axis-aligned segments)."""
    pts = np.asarray(waypoints, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.abs(np.diff(pts, axis=0)).sum())


def locations_to_points(layout, locations_df):
    """Map a DataFrame with x, y columns to a list of PickPoints."""
    return [layout.to_pick_point(x, y) for x, y in zip(locations_df["x"], locations_df["y"])]


def layout_summary(layout):
    """Readable summary used in the report and the UI."""
    return pd.DataFrame([
        {"item": "Picking aisles (left block)", "value": len(layout.aisle_ys)},
        {"item": "Picking aisles (right block)", "value": len(layout.right_aisle_ys)},
        {"item": "Cross aisles", "value": 3},
        {"item": "Left aisle length (m)", "value": round(layout.aisle_length["L"] * config.UNIT_TO_METRES, 2)},
        {"item": "Right aisle length (m)", "value": round(layout.aisle_length["R"] * config.UNIT_TO_METRES, 2)},
        {"item": "Depot (x, y)", "value": str(layout.depot_xy)},
    ])
