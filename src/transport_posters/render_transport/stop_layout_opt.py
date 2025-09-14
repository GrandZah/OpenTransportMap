from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
import math
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RectPX:
    l: float
    b: float
    r: float
    t: float

    def intersects(self, o: "RectPX") -> bool:
        return not (self.r <= o.l or self.l >= o.r or self.t <= o.b or self.b >= o.t)

    def intersection_area(self, o: "RectPX") -> float:
        if not self.intersects(o):
            return 0.0
        w = max(0.0, min(self.r, o.r) - max(self.l, o.l))
        h = max(0.0, min(self.t, o.t) - max(self.b, o.b))
        return w * h

    @staticmethod
    def union(a: "RectPX", b: "RectPX") -> "RectPX":
        return RectPX(min(a.l, b.l), min(a.b, b.b), max(a.r, b.r), max(a.t, b.t))


@dataclass
class CandidatePair:
    stop_idx: int

    name_offset_pt: Tuple[float, float]
    routes_text_offset_pt: Tuple[float, float]

    name_rect_px: RectPX
    routes_box_rect_px: RectPX
    corridor_rect_px: RectPX
    pair_union_rect_px: RectPX

    side_sign: int
    is_near_edge: bool
    point_px: Tuple[float, float]
    stop_radius_px: float

    lines_used: int
    anchor_dist_px: float
    routes_dist_px: float


@dataclass
class Precomp:
    local_costs: List[List[float]]

    pair_costs: Dict[Tuple[int, int], List[List[float]]]

    cand_counts: List[int]


def _occlude_penalty(px: float, py: float, r_px: float, rect: RectPX) -> float:
    """Thickness of rectangle penetration into circle radius *r_px* around (px, py)."""
    dx = 0.0 if rect.l <= px <= rect.r else (rect.l - px if px < rect.l else px - rect.r)
    dy = 0.0 if rect.b <= py <= rect.t else (rect.b - py if py < rect.b else py - rect.t)
    d = math.hypot(dx, dy)
    return max(0.0, r_px - d)


def cand_local_cost(c: CandidatePair, cfg) -> float:
    occ = (_occlude_penalty(c.point_px[0], c.point_px[1], c.stop_radius_px, c.name_rect_px) +
           _occlude_penalty(c.point_px[0], c.point_px[1], c.stop_radius_px, c.routes_box_rect_px))

    name_w_px = max(1.0, c.name_rect_px.r - c.name_rect_px.l)

    len_factor_raw = 1.0 + cfg.ROUTES_FAR_LEN_BOOST * (name_w_px / cfg.NAME_WIDTH_REF_PX)
    len_factor = min(cfg.ROUTES_NEAR_FACTOR_MAX, len_factor_raw)

    routes_term = cfg.W_ROUTES_NEAR * len_factor * c.routes_dist_px

    return (cfg.W_OFFSET * c.anchor_dist_px +
            cfg.W_LINES * max(0, c.lines_used - 1) +
            routes_term +
            cfg.W_OCCLUDE_STOP * occ)


def pair_cost(a: CandidatePair, b: CandidatePair, cfg) -> float:
    if not a.pair_union_rect_px.intersects(b.pair_union_rect_px):
        ca = corr = 0.0
    else:
        ca = (a.name_rect_px.intersection_area(b.name_rect_px) +
              a.name_rect_px.intersection_area(b.routes_box_rect_px) +
              a.routes_box_rect_px.intersection_area(b.name_rect_px) +
              a.routes_box_rect_px.intersection_area(b.routes_box_rect_px))
        corr = (a.corridor_rect_px.intersection_area(b.name_rect_px) +
                a.corridor_rect_px.intersection_area(b.routes_box_rect_px) +
                b.corridor_rect_px.intersection_area(a.name_rect_px) +
                b.corridor_rect_px.intersection_area(a.routes_box_rect_px))

    occ = (
            _occlude_penalty(b.point_px[0], b.point_px[1], b.stop_radius_px, a.name_rect_px) +
            _occlude_penalty(b.point_px[0], b.point_px[1], b.stop_radius_px, a.routes_box_rect_px) +
            _occlude_penalty(a.point_px[0], a.point_px[1], a.stop_radius_px, b.name_rect_px) +
            _occlude_penalty(a.point_px[0], a.point_px[1], a.stop_radius_px, b.routes_box_rect_px)
    )

    dx = a.point_px[0] - b.point_px[0]
    dy = a.point_px[1] - b.point_px[1]
    d = math.hypot(dx, dy)
    coh = 0.0
    R = cfg.SIDE_NEIGHBOR_RADIUS_PX
    if d < R and (a.side_sign != b.side_sign):
        weight = 1.0 - d / R
        coh = weight
    return (cfg.W_OVERLAP * ca +
            cfg.W_CORRIDOR * corr +
            cfg.W_OCCLUDE_STOP * occ +
            cfg.W_SIDE_COHESION * coh)


def precompute_costs(per_stop_cands: List[List[CandidatePair]], cfg) -> Precomp:
    """Precompute local and pairwise costs for candidate pairs."""
    n = len(per_stop_cands)
    local_costs: List[List[float]] = []
    for i, cands in enumerate(per_stop_cands):
        local_costs.append([cand_local_cost(c, cfg) for c in cands])

    pair_costs: Dict[Tuple[int, int], List[List[float]]] = {}
    for i in range(n):
        Mi = len(per_stop_cands[i])
        for j in range(i + 1, n):
            Mj = len(per_stop_cands[j])
            mat = [[0.0] * Mj for _ in range(Mi)]
            ai = per_stop_cands[i]
            bj = per_stop_cands[j]
            for ci in range(Mi):
                for cj in range(Mj):
                    mat[ci][cj] = pair_cost(ai[ci], bj[cj], cfg)
            pair_costs[(i, j)] = mat
    return Precomp(local_costs=local_costs,
                   pair_costs=pair_costs,
                   cand_counts=[len(cs) for cs in per_stop_cands])


def _pair_cost_lookup(pre: Precomp, i: int, ci: int, j: int, cj: int) -> float:
    if i == j:
        return 0.0
    if i < j:
        return pre.pair_costs[(i, j)][ci][cj]
    else:
        return pre.pair_costs[(j, i)][cj][ci]


def total_cost(selection: List[int], pre: Precomp) -> float:
    """Compute total cost for a given selection."""
    n = len(selection)
    s = 0.0
    for i in range(n):
        s += pre.local_costs[i][selection[i]]
    for i in range(n):
        ci = selection[i]
        for j in range(i + 1, n):
            cj = selection[j]
            s += pre.pair_costs[(i, j)][ci][cj]
    return s


def delta_cost_for_changes(selection: List[int],
                           pre: Precomp,
                           changed: Dict[int, int]) -> float:
    """Compute cost delta if indices in *changed* switch to new candidates."""
    n = len(selection)
    delta = 0.0

    for i, new_ci in changed.items():
        old_ci = selection[i]
        delta += pre.local_costs[i][new_ci] - pre.local_costs[i][old_ci]

    changed_keys = list(changed.keys())
    changed_set = set(changed_keys)
    for i in changed_keys:
        new_ci = changed[i]
        old_ci = selection[i]
        for j in range(n):
            if j == i or j in changed_set:
                continue
            cj = selection[j]
            delta += _pair_cost_lookup(pre, i, new_ci, j, cj) - _pair_cost_lookup(pre, i, old_ci, j, cj)

    ck = len(changed_keys)
    for a in range(ck):
        i = changed_keys[a]
        new_ci = changed[i]
        old_ci = selection[i]
        for b in range(a + 1, ck):
            j = changed_keys[b]
            new_cj = changed[j]
            old_cj = selection[j]
            delta += _pair_cost_lookup(pre, i, new_ci, j, new_cj) - _pair_cost_lookup(pre, i, old_ci, j, old_cj)
    return delta


def build_clusters(per_stop_cands: List[List[CandidatePair]],
                   pad_px: float = 4.0) -> List[List[int]]:
    """Build clusters by intersection of stop "shells".
       A shell is the union of pair_union_rect_px for all candidates, expanded by pad_px."""
    n = len(per_stop_cands)
    shells: List[RectPX] = []
    for i, cands in enumerate(per_stop_cands):
        if not cands:
            shells.append(RectPX(0, 0, 0, 0))
            continue
        u = cands[0].pair_union_rect_px
        l, b, r, t = u.l, u.b, u.r, u.t
        for k in range(1, len(cands)):
            u2 = cands[k].pair_union_rect_px
            l = min(l, u2.l)
            b = min(b, u2.b)
            r = max(r, u2.r)
            t = max(t, u2.t)
        shells.append(RectPX(l - pad_px, b - pad_px, r + pad_px, t + pad_px))

    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        si = shells[i]
        for j in range(i + 1, n):
            if si.intersects(shells[j]):
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    clusters: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        comp: List[int] = []
        seen[i] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        clusters.append(sorted(comp))

    clusters.sort(key=len, reverse=True)
    return clusters


def _hot_spots(selection: List[int], pre: Precomp, cluster: Iterable[int],
               cfg) -> List[int]:
    scores = []
    cluster_set = set(cluster)
    for i in cluster:

        loc = pre.local_costs[i][selection[i]]
        ov = 0.0
        for j in range(len(selection)):
            if j == i:
                continue
            ov += _pair_cost_lookup(pre, i, selection[i], j, selection[j])
        scores.append((loc + ov, i))
    scores.sort(reverse=True)
    k2 = min(cfg.HOTK_2OPT, len(scores))
    return [i for _, i in scores[:k2]]


def hill_climb_cluster(
        selection: List[int],
        per_stop_cands: List[List[CandidatePair]],
        pre: Precomp,
        cluster: Iterable[int],
        time_budget_sec: float,
        cfg,
        debug: bool = True,
        max_iters: int = 20_000,
) -> float:
    """Single-point flips within a cluster, modifying selection in-place.
    Returns new total cost (for reporting)."""
    t0 = time.perf_counter()
    cost = total_cost(selection, pre)

    it = 0
    improved_any = True
    changes = 0
    while improved_any and (time.perf_counter() - t0) < time_budget_sec:
        it += 1
        improved_any = False
        for i in cluster:
            old_ci = selection[i]
            Mi = pre.cand_counts[i]
            best_ci = old_ci
            best_delta = 0.0

            for ci in range(Mi):
                if ci == old_ci:
                    continue
                d = delta_cost_for_changes(selection, pre, {i: ci})

                if (d + 1e-6 < best_delta) or (
                        abs(d - best_delta) <= 1e-6
                        and per_stop_cands[i][ci].is_near_edge
                        and not per_stop_cands[i][best_ci].is_near_edge
                ):
                    best_delta = d
                    best_ci = ci
                if (time.perf_counter() - t0) >= time_budget_sec:
                    break
            if best_ci != old_ci and (best_delta < -1e-9 or abs(best_delta) <= 1e-6):
                selection[i] = best_ci
                cost += best_delta
                improved_any = True
                changes += 1
            if (time.perf_counter() - t0) >= time_budget_sec:
                break
        if it >= max_iters:
            break

    if debug:
        reason = (
            "time_limit"
            if (time.perf_counter() - t0) >= time_budget_sec
            else ("max_iters" if it >= max_iters else "local_optimum")
        )
        logger.info(
            f"cluster hill-climb: reason={reason}, iters={it}, changes={changes}, cost={cost:.3f}"
        )
    return cost


def two_opt_cluster(
        selection: List[int],
        per_stop_cands: List[List[CandidatePair]],
        pre: Precomp,
        cluster: Iterable[int],
        time_budget_sec: float,
        cfg,
        debug: bool = True,
) -> float:
    """2-opt on hot nodes. Modifies selection in-place and returns new cost."""
    t0 = time.perf_counter()
    cost = total_cost(selection, pre)
    accepted2 = 0

    if time_budget_sec <= 0.0:
        if debug:
            logger.info("cluster multi-opt: accepts2=0, cost=%.3f, time_spent=0.000s", cost)
        return cost

    hot = _hot_spots(selection, pre, cluster, cfg)

    stop = False
    for a in range(len(hot)):
        i = hot[a]
        for b in range(a + 1, len(hot)):
            j = hot[b]
            cur_i, cur_j = selection[i], selection[j]
            Mi, Mj = pre.cand_counts[i], pre.cand_counts[j]
            best_pair_delta = 0.0
            best_ki, best_kj = cur_i, cur_j
            for ki in range(Mi):
                if ki == cur_i:
                    continue
                for kj in range(Mj):
                    if kj == cur_j:
                        continue
                    d = delta_cost_for_changes(selection, pre, {i: ki, j: kj})
                    better_by_near = (
                            (per_stop_cands[i][ki].is_near_edge + per_stop_cands[j][kj].is_near_edge)
                            > (per_stop_cands[i][cur_i].is_near_edge + per_stop_cands[j][cur_j].is_near_edge)
                    )
                    if (d + 1e-6 < best_pair_delta) or (
                            abs(d - best_pair_delta) <= 1e-6 and better_by_near
                    ):
                        best_pair_delta = d
                        best_ki, best_kj = ki, kj
                    if (time.perf_counter() - t0) >= time_budget_sec:
                        stop = True
                        break
                if stop:
                    break
            if best_ki != cur_i or best_kj != cur_j:
                selection[i] = best_ki
                selection[j] = best_kj
                cost += best_pair_delta
                accepted2 += 1
                if debug:
                    logger.debug(
                        f"cluster 2-opt: i={i}->{best_ki} j={j}->{best_kj} Δ={best_pair_delta:.2f} cost={cost:.2f}"
                    )
            if stop:
                break
        if stop:
            break

    if debug:
        logger.info(
            "cluster multi-opt: accepts2=%d, cost=%.3f, time_spent=%.3fs",
            accepted2,
            cost,
            (time.perf_counter() - t0),
        )
    return cost


def optimize_cluster(
        selection: List[int],
        per_stop_cands: List[List[CandidatePair]],
        pre: Precomp,
        cluster: List[int],
        time_budget_sec: float,
        cfg,
        debug: bool = True,
) -> None:
    """Iteratively improve selection in-place for indexes from cluster.
    Two cycles (HILL → 2-OPT) and final HILL polish."""
    if debug:
        logger.info(
            f"cluster start: size={len(cluster)}, budget={time_budget_sec:.3f}s"
        )

    t0 = time.perf_counter()

    FRAC_H1 = 0.35
    FRAC_O1 = 0.25
    FRAC_H2 = 0.20
    FRAC_O2 = 0.10

    left = max(0.0, time_budget_sec - (time.perf_counter() - t0))
    _ = hill_climb_cluster(
        selection, per_stop_cands, pre, cluster,
        min(FRAC_H1 * time_budget_sec, left), cfg, debug
    )

    left = max(0.0, time_budget_sec - (time.perf_counter() - t0))
    _ = two_opt_cluster(
        selection, per_stop_cands, pre, cluster,
        min(FRAC_O1 * time_budget_sec, left), cfg, debug
    )

    left = max(0.0, time_budget_sec - (time.perf_counter() - t0))
    _ = hill_climb_cluster(
        selection, per_stop_cands, pre, cluster,
        min(FRAC_H2 * time_budget_sec, left), cfg, debug
    )

    left = max(0.0, time_budget_sec - (time.perf_counter() - t0))
    _ = two_opt_cluster(
        selection, per_stop_cands, pre, cluster,
        min(FRAC_O2 * time_budget_sec, left), cfg, debug
    )

    left = max(0.0, time_budget_sec - (time.perf_counter() - t0))
    T_final_min = 0.05 * time_budget_sec
    if left < T_final_min:
        left = T_final_min
    _ = hill_climb_cluster(
        selection, per_stop_cands, pre, cluster,
        left, cfg, debug
    )


def cost_breakdown_per_stop(selection: List[int],
                            per_stop_cands: List[List[CandidatePair]],
                            pre: Precomp,
                            cfg) -> List[Tuple[float, float, float, float, float]]:
    """For each i return (local, overlap, corridor, occ, cohesion) already weighted."""
    n = len(selection)

    def decompose_pair(a: CandidatePair, b: CandidatePair) -> Tuple[float, float, float, float]:

        if not a.pair_union_rect_px.intersects(b.pair_union_rect_px):
            ca = corr = 0.0
        else:
            ca = (a.name_rect_px.intersection_area(b.name_rect_px) +
                  a.name_rect_px.intersection_area(b.routes_box_rect_px) +
                  a.routes_box_rect_px.intersection_area(b.name_rect_px) +
                  a.routes_box_rect_px.intersection_area(b.routes_box_rect_px))
            corr = (a.corridor_rect_px.intersection_area(b.name_rect_px) +
                    a.corridor_rect_px.intersection_area(b.routes_box_rect_px) +
                    b.corridor_rect_px.intersection_area(a.name_rect_px) +
                    b.corridor_rect_px.intersection_area(a.routes_box_rect_px))
        occ = (
                _occlude_penalty(b.point_px[0], b.point_px[1], b.stop_radius_px, a.name_rect_px) +
                _occlude_penalty(b.point_px[0], b.point_px[1], b.stop_radius_px, a.routes_box_rect_px) +
                _occlude_penalty(a.point_px[0], a.point_px[1], a.stop_radius_px, b.name_rect_px) +
                _occlude_penalty(a.point_px[0], a.point_px[1], a.stop_radius_px, b.routes_box_rect_px)
        )
        dx = a.point_px[0] - b.point_px[0]
        dy = a.point_px[1] - b.point_px[1]
        d = math.hypot(dx, dy)
        coh = 0.0
        R = cfg.SIDE_NEIGHBOR_RADIUS_PX
        if d < R and (a.side_sign != b.side_sign):
            coh = 1.0 - d / R
        return (cfg.W_OVERLAP * ca, cfg.W_CORRIDOR * corr, cfg.W_OCCLUDE_STOP * occ, cfg.W_SIDE_COHESION * coh)

    out = []
    chosen = [per_stop_cands[i][selection[i]] for i in range(n)]
    for i in range(n):
        loc = pre.local_costs[i][selection[i]]
        ov = cr = oc = co = 0.0
        for j in range(n):
            if i == j:
                continue
            a, b = chosen[i], chosen[j]
            ovi, cri, oci, coi = decompose_pair(a, b)
            ov += ovi
            cr += cri
            oc += oci
            co += coi
        out.append((loc, ov, cr, oc, co))
    return out


def optimize_all(per_stop_cands: List[List[CandidatePair]],
                 pre: Precomp,
                 time_budget_sec: float,
                 cfg,
                 debug: bool = True) -> List[int]:
    """Optimise candidate selection for all stops within time budget."""
    n = len(per_stop_cands)

    selection = [0] * n
    for i, cands in enumerate(per_stop_cands):
        lc = pre.local_costs[i]
        selection[i] = min(range(len(cands)), key=lambda k: lc[k])

    clusters = build_clusters(per_stop_cands, pad_px=4.0)
    if debug:
        sizes = [len(c) for c in clusters]
        logger.info(f"clusters: count={len(clusters)} sizes={sizes}")

    total_nodes = sum(len(c) for c in clusters) or 1
    t0 = time.perf_counter()
    for idx, cluster in enumerate(clusters):
        spent = time.perf_counter() - t0
        left = max(0.0, time_budget_sec - spent)
        if left <= 0.0:
            if debug:
                logger.info("time budget exhausted before finishing clusters")
            break
        share = max(0.05, left * (len(cluster) / total_nodes))
        optimize_cluster(selection, per_stop_cands, pre, cluster, share, cfg, debug=debug)

    return selection
