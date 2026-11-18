"""PERSON 4 - HMM + Viterbi route reconstruction.

Hidden state at time i  = one candidate arc of GPS observation i
                           (state keeps arc_id and position along the arc -> candidate/arc mapping preserved)
Emission  log P(z_i | arc)   = w_e * log(classifier probability of that candidate)
Transition log P(b | a)      = -| network_distance(a -> b) - gps_distance(i-1, i) | / beta
    network_distance: same arc -> distance along arc; a.v == b.u -> rest of a + start of b;
                      otherwise shortest path (Dijkstra) between a.v and b.u + partial arcs.
    unreachable within `max_route_m`  -> NO_PATH_LOGP (very unlikely but not impossible, so decoding never breaks)
    GPS gap (dt > gap_s or jump > gap_m) -> transition reset (all transitions equal), decoder relies on emissions
Decoding: standard Viterbi in log space with back-pointers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DEFAULT_HMM = dict(beta=5.0, w_emission=1.0, max_route_m=1500.0, no_path_logp=-30.0,
                   gap_s=10.0, gap_m=500.0, p_floor=1e-4)


class RoadGraph:
    """Directed road graph restricted to arcs near the trajectory, with cached Dijkstra."""

    def __init__(self, trip, pad_m=1500.0):
        px, py = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()
        ax, ay, bx, by = trip.arc_xy()
        self.L = np.hypot(bx - ax, by - ay)
        self.u, self.v = trip.arc_u, trip.arc_v
        near = ((np.minimum(ax, bx) < px.max() + pad_m) & (np.maximum(ax, bx) > px.min() - pad_m) &
                (np.minimum(ay, by) < py.max() + pad_m) & (np.maximum(ay, by) > py.min() - pad_m))
        ids = np.flatnonzero(near)
        nodes = np.unique(np.r_[self.u[ids], self.v[ids]])
        self.local = {n: i for i, n in enumerate(nodes)}
        self.nodes = nodes
        lu = np.searchsorted(nodes, self.u[ids])
        lv = np.searchsorted(nodes, self.v[ids])
        w = np.maximum(self.L[ids], 1e-3)  # zero-length arcs would vanish from sparse matrix
        G = pd.DataFrame(dict(a=lu, b=lv, w=w)).groupby(["a", "b"], as_index=False).w.min()
        self.G = csr_matrix((G.w, (G.a, G.b)), shape=(len(nodes), len(nodes)))
        self.cache = {}
        # (u,v) -> shortest arc id, to turn node paths back into arcs
        df = pd.DataFrame(dict(u=self.u[ids], v=self.v[ids], arc=ids, L=self.L[ids])).sort_values("L")
        self.arc_of = {(a, b): c for a, b, c in df.drop_duplicates(["u", "v"])[["u", "v", "arc"]].itertuples(index=False)}

    def dist(self, src, dst, limit):
        """Shortest path length src->dst (global node ids) or inf if not within limit."""
        if src not in self.local or dst not in self.local:
            return np.inf
        if src not in self.cache:
            d = dijkstra(self.G, indices=self.local[src], limit=limit)
            ok = np.flatnonzero(np.isfinite(d))
            self.cache[src] = dict(zip(ok.tolist(), d[ok].tolist()))
        return self.cache[src].get(self.local[dst], np.inf)

    def path_arcs(self, src, dst):
        """Arc ids of shortest path src->dst (for drawing a continuous route)."""
        if src not in self.local or dst not in self.local:
            return []
        _, pred = dijkstra(self.G, indices=self.local[src], return_predecessors=True, limit=5000)
        j, s = self.local[dst], self.local[src]
        if pred[j] < 0 and j != s:
            return []
        seq = [j]
        while seq[-1] != s:
            seq.append(pred[seq[-1]])
        seq = [self.nodes[k] for k in seq[::-1]]
        return [self.arc_of[(a, b)] for a, b in zip(seq[:-1], seq[1:]) if (a, b) in self.arc_of]


def network_distance(g, a, fa, b, fb, limit):
    if a == b:
        return abs(fb - fa) * g.L[a]
    rest_a, start_b = (1 - fa) * g.L[a], fb * g.L[b]
    if g.v[a] == g.u[b]:
        return rest_a + start_b
    sp = g.dist(g.v[a], g.u[b], limit)
    return rest_a + sp + start_b


def viterbi(trip, cand, params=None, graph=None):
    """cand: DataFrame with obs_idx, arc_id, p, proj_frac for ONE trip.
    Returns array of predicted arc_id per observation index present in cand (sorted by obs_idx)."""
    P = {**DEFAULT_HMM, **(params or {})}
    g = graph or RoadGraph(trip)
    px, py, ts = (trip.track[c].to_numpy() for c in ("x", "y", "t"))
    cand = cand.sort_values(["obs_idx", "p"], ascending=[True, False])
    groups = [(i, grp["arc_id"].to_numpy(), grp["proj_frac"].to_numpy(), grp["p"].to_numpy())
              for i, grp in cand.groupby("obs_idx", sort=True)]

    obs_ids = [x[0] for x in groups]
    emis = [P["w_emission"] * np.log(np.clip(x[3], P["p_floor"], 1.0)) for x in groups]
    score = emis[0].copy()
    back = [None]
    for k in range(1, len(groups)):
        i0, A, FA, _ = groups[k - 1]
        i1, B, FB, _ = groups[k]
        dt = ts[i1] - ts[i0]
        gps_d = float(np.hypot(px[i1] - px[i0], py[i1] - py[i0]))
        if dt > P["gap_s"] or gps_d > P["gap_m"]:
            T = np.zeros((len(A), len(B)))  # gap: reset transitions
        else:
            beta = P["beta"] * max(1.0, dt)
            limit = max(P["max_route_m"], 3 * gps_d)
            T = np.empty((len(A), len(B)))
            for x in range(len(A)):
                for y in range(len(B)):
                    nd = network_distance(g, A[x], FA[x], B[y], FB[y], limit)
                    T[x, y] = P["no_path_logp"] if not np.isfinite(nd) else -abs(nd - gps_d) / beta
        tot = score[:, None] + T
        bp = tot.argmax(0)
        score = tot[bp, np.arange(len(B))] + emis[k]
        back.append(bp)

    # backtrack
    path = np.empty(len(groups), dtype=np.int64)
    j = int(score.argmax())
    for k in range(len(groups) - 1, -1, -1):
        path[k] = groups[k][1][j]
        if k > 0:
            j = int(back[k][j])
    return np.asarray(obs_ids), path


def continuous_route(g, arcs):
    """Collapse per-observation arcs into a route and fill non-connected jumps with shortest paths."""
    seq = [a for i, a in enumerate(arcs) if i == 0 or a != arcs[i - 1]]
    out = []
    for a in seq:
        if out and g.v[out[-1]] != g.u[a]:
            out += g.path_arcs(g.v[out[-1]], g.u[a])
        out.append(int(a))
    return out
