"""Publication-quality plasmid map and composition plots for PlasmidQC-Studio.

The single public entry point, :func:`plot_plasmid_dashboard`, renders a
two-panel 300-DPI figure:

* **left** -- a circular plasmid map with staggered feature arcs, non-overlapping
  radial labels joined by leader lines, and internal tick markers for unique
  restriction cutters;
* **right** -- a dual-axis composition profile showing sliding-window GC% and
  cumulative GC-skew.

Matplotlib is used with the ``Agg`` backend so the module is safe to import in
headless CI and inside Streamlit's worker threads.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow, Wedge

from .analyzer import AnnotatedFeature, RestrictionHit, gc_content, sliding_gc_skew

__all__ = ["plot_plasmid_dashboard"]

DPI = 300
_BACKBONE_R = 1.0
_TRACK_WIDTH = 0.075
_TRACK_GAP = 0.016
_LABEL_R = 1.30
_ENZYME_R = 0.845


def _bp_to_theta(pos_bp: float, total_bp: int) -> float:
    """Map a 1-based coordinate to a map angle in degrees.

    Position 1 sits at 12 o'clock and coordinates increase clockwise, matching
    the convention used by SnapGene and Benchling.
    """
    return 90.0 - 360.0 * (pos_bp - 1) / total_bp


def _feature_span(feat: AnnotatedFeature, total_bp: int) -> Tuple[float, float]:
    """Return ``(theta_start, theta_end)`` in degrees, handling origin wrap."""
    start_theta = _bp_to_theta(feat.start, total_bp)
    span_deg = 360.0 * feat.length_bp / total_bp
    return start_theta - span_deg, start_theta


def _assign_tracks(features: Sequence[AnnotatedFeature],
                   total_bp: int) -> List[int]:
    """Greedy interval-graph colouring to stagger overlapping features.

    Features are placed on the innermost radial track that is free at their
    coordinates, so overlapping elements (an RBS inside a UTR, a promoter
    abutting a CDS) stay individually visible instead of being overdrawn.
    """
    occupied: List[List[Tuple[float, float]]] = []
    tracks: List[int] = []
    pad = 360.0 * 0.004

    for feat in features:
        start = 360.0 * (feat.start - 1) / total_bp
        end = start + 360.0 * feat.length_bp / total_bp
        intervals = [(start - pad, end + pad)]
        if end > 360.0:                     # wraps the origin
            intervals = [(start - pad, 360.0), (0.0, end - 360.0 + pad)]

        placed = None
        for t, spans in enumerate(occupied):
            if all(not (a < e2 and s2 < b) for s2, e2 in spans
                   for a, b in intervals):
                placed = t
                break
        if placed is None:
            placed = len(occupied)
            occupied.append([])
        occupied[placed].extend(intervals)
        tracks.append(placed)
    return tracks


def _spread_labels(angles: Sequence[float], min_sep: float = 11.0
                   ) -> List[float]:
    """Push label angles apart so text does not collide.

    Labels are separated independently on the right and left halves of the map
    (text is anchored outward on each side), preserving their original vertical
    order while enforcing ``min_sep`` degrees of separation.
    """
    out = list(angles)
    for side in ("right", "left"):
        idx = [i for i, a in enumerate(angles)
               if (math.cos(math.radians(a)) >= 0) == (side == "right")]
        if not idx:
            continue
        idx.sort(key=lambda i: angles[i])
        vals = [angles[i] for i in idx]
        for _ in range(240):
            moved = False
            for k in range(len(vals) - 1):
                gap = vals[k + 1] - vals[k]
                if gap < min_sep:
                    shift = (min_sep - gap) / 2.0
                    vals[k] -= shift
                    vals[k + 1] += shift
                    moved = True
            if not moved:
                break
        for i, v in zip(idx, vals):
            out[i] = v
    return out


def _draw_circular_map(ax, plasmid_name: str, seq: str,
                       features: Sequence[AnnotatedFeature],
                       restriction_hits: Sequence[RestrictionHit]) -> None:
    """Render the circular backbone, feature arcs, labels, and cutter ticks."""
    total_bp = len(seq)

    ax.set_aspect("equal")
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.62, 1.62)
    ax.axis("off")

    # Backbone ring.
    ax.add_patch(Circle((0, 0), _BACKBONE_R, fill=False, lw=2.1,
                        edgecolor="#2b2b2b", zorder=2))
    ax.add_patch(Circle((0, 0), _BACKBONE_R, fill=False, lw=6.0,
                        edgecolor="#2b2b2b", alpha=0.10, zorder=1))

    # Coordinate ruler: major ticks every 500 bp, minor every 100 bp.
    for bp in range(0, total_bp, 100):
        th = math.radians(_bp_to_theta(bp + 1, total_bp))
        major = bp % 500 == 0
        r0 = _BACKBONE_R
        r1 = _BACKBONE_R + (0.040 if major else 0.020)
        ax.plot([r0 * math.cos(th), r1 * math.cos(th)],
                [r0 * math.sin(th), r1 * math.sin(th)],
                color="#4d4d4d", lw=1.0 if major else 0.6, zorder=3)
        if major:
            rt = _BACKBONE_R + 0.085
            ax.text(rt * math.cos(th), rt * math.sin(th), f"{bp:,}",
                    ha="center", va="center", fontsize=5.6, color="#4d4d4d",
                    zorder=3)

    # Feature arcs, staggered onto radial tracks.
    tracks = _assign_tracks(features, total_bp)
    label_specs: List[Tuple[float, float, str, str]] = []

    for feat, track in zip(features, tracks):
        r_out = _BACKBONE_R - _TRACK_GAP - track * (_TRACK_WIDTH + _TRACK_GAP)
        r_in = r_out - _TRACK_WIDTH
        th1, th2 = _feature_span(feat, total_bp)
        ax.add_patch(Wedge((0, 0), r_out, th1, th2, width=_TRACK_WIDTH,
                           facecolor=feat.color, edgecolor="white", lw=0.45,
                           alpha=0.93, zorder=4))

        # Directional arrowhead at the feature's 3' end.
        if feat.length_bp > total_bp * 0.012:
            tip_deg = th1 if feat.strand == "-" else th2
            tip = math.radians(tip_deg)
            back = math.radians(tip_deg + (4.2 if feat.strand == "-" else -4.2))
            r_mid = (r_in + r_out) / 2.0
            ax.add_patch(FancyArrow(
                r_mid * math.cos(back), r_mid * math.sin(back),
                r_mid * math.cos(tip) - r_mid * math.cos(back),
                r_mid * math.sin(tip) - r_mid * math.sin(back),
                width=0.0, head_width=_TRACK_WIDTH * 0.92,
                head_length=0.052, length_includes_head=True,
                facecolor=feat.color, edgecolor="white", lw=0.3, zorder=5))

        mid_deg = (th1 + th2) / 2.0
        label = f"{feat.name} ({feat.length_bp:,} bp, {feat.strand})"
        label_specs.append((mid_deg, r_out, label, feat.color))

    # Non-overlapping outer labels with leader lines.
    spread = _spread_labels([m for m, _, _, _ in label_specs])
    for (mid_deg, r_out, label, color), lab_deg in zip(label_specs, spread):
        th_f = math.radians(mid_deg)
        th_l = math.radians(lab_deg)
        r_anchor = r_out + 0.012
        x0, y0 = r_anchor * math.cos(th_f), r_anchor * math.sin(th_f)
        x1, y1 = (_LABEL_R - 0.10) * math.cos(th_l), (_LABEL_R - 0.10) * math.sin(th_l)
        x2, y2 = _LABEL_R * math.cos(th_l), _LABEL_R * math.sin(th_l)
        ax.add_line(Line2D([x0, x1, x2], [y0, y1, y2], color=color, lw=0.7,
                           alpha=0.8, zorder=3, solid_capstyle="round"))
        right = math.cos(th_l) >= 0
        ax.text(x2 + (0.028 if right else -0.028), y2, label,
                ha="left" if right else "right", va="center",
                fontsize=6.4, color="#1a1a1a", zorder=6)

    # Internal radial markers for unique restriction cutters.
    # Labels are written along the radius (SnapGene convention) rather than
    # tangentially, so they read inward from their tick without crossing the
    # feature arcs or each other.
    unique = [h for h in restriction_hits if h.n_sites == 1]
    for hit in unique:
        pos = hit.positions[0]
        deg = _bp_to_theta(pos, total_bp)
        th = math.radians(deg)
        ax.plot([_ENZYME_R * math.cos(th), (_ENZYME_R - 0.055) * math.cos(th)],
                [_ENZYME_R * math.sin(th), (_ENZYME_R - 0.055) * math.sin(th)],
                color="#111111", lw=0.9, zorder=7)
        rt = _ENZYME_R - 0.075
        rot = deg % 360
        right = math.cos(th) >= 0
        if not right:
            rot -= 180
        ax.text(rt * math.cos(th), rt * math.sin(th),
                f"{hit.enzyme} {pos:,}",
                ha="right" if right else "left", va="center",
                fontsize=5.8, color="#111111", rotation=rot,
                rotation_mode="anchor", zorder=7)

    # Hub annotation.
    ax.text(0, 0.085, plasmid_name, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1a1a1a")
    ax.text(0, -0.035, f"{total_bp:,} bp  |  circular", ha="center",
            va="center", fontsize=7.6, color="#4d4d4d")
    ax.text(0, -0.135, f"GC {gc_content(seq):.1f}%  |  "
                       f"{len(unique)} unique cutters",
            ha="center", va="center", fontsize=7.0, color="#4d4d4d")

    # Legend by feature class.
    seen = {}
    for feat in features:
        seen.setdefault(feat.feature_type, feat.color)
    handles = [Line2D([], [], color=c, lw=5.2, solid_capstyle="butt", label=t)
               for t, c in sorted(seen.items())]
    handles.append(Line2D([], [], color="#111111", lw=0.9,
                          label="unique cutter"))
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.075), ncol=min(4, len(handles)),
              frameon=False, fontsize=6.3, handlelength=1.5,
              columnspacing=1.1, handletextpad=0.5)


def _draw_composition(ax, seq: str, window: int, step: int) -> None:
    """Render the dual-axis GC% / cumulative GC-skew profile."""
    positions, gc, _skew, cumulative = sliding_gc_skew(seq, window=window,
                                                       step=step)
    x = np.arange(len(positions)) * step + window / 2.0
    gc = np.asarray(gc)
    mean_gc = float(gc.mean()) if len(gc) else 0.0

    ax.plot(x, gc, color="#1f6fb4", lw=1.15, zorder=3,
            label=f"GC% ({window} bp window)")
    ax.fill_between(x, mean_gc, gc, where=gc >= mean_gc, color="#1f6fb4",
                    alpha=0.16, zorder=2, linewidth=0)
    ax.fill_between(x, mean_gc, gc, where=gc < mean_gc, color="#f08c3c",
                    alpha=0.16, zorder=2, linewidth=0)
    ax.axhline(mean_gc, color="#1f6fb4", lw=0.8, ls="--", alpha=0.65,
               zorder=3, label=f"mean GC {mean_gc:.1f}%")

    ax.set_xlabel("Position (bp)", fontsize=8.5)
    ax.set_ylabel("GC content (%)", fontsize=8.5, color="#1f6fb4")
    ax.tick_params(axis="y", labelcolor="#1f6fb4", labelsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlim(0, len(seq))
    lo, hi = (gc.min(), gc.max()) if len(gc) else (0.0, 100.0)
    pad = max(2.0, (hi - lo) * 0.22)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, axis="both", alpha=0.18, lw=0.5)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(x, cumulative, color="#8c3bbf", lw=1.3, zorder=4,
             label="cumulative GC-skew")
    ax2.axhline(0.0, color="#8c3bbf", lw=0.7, ls=":", alpha=0.6, zorder=3)
    ax2.set_ylabel("Cumulative GC-skew  (G-C)/(G+C)", fontsize=8.5,
                   color="#8c3bbf")
    ax2.tick_params(axis="y", labelcolor="#8c3bbf", labelsize=7.5)
    ax2.spines["top"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc="upper right", fontsize=6.6,
              frameon=True, framealpha=0.9, edgecolor="#d9d9d9")
    ax.set_title("Composition profile (circular, wrap-around windows)",
                 fontsize=9.5, pad=14)


def plot_plasmid_dashboard(plasmid_name: str,
                           seq: str,
                           features: Iterable[AnnotatedFeature],
                           restriction_hits: Iterable[RestrictionHit],
                           output_path: Optional[str] = None,
                           window: int = 120,
                           step: int = 20,
                           dpi: int = DPI):
    """Render the two-panel plasmid QC dashboard at 300 DPI.

    Parameters
    ----------
    plasmid_name : str
        Title shown at the hub of the circular map.
    seq : str
        Plasmid sequence.
    features : iterable of AnnotatedFeature
        Output of :func:`src.analyzer.annotate_plasmid`.
    restriction_hits : iterable of RestrictionHit
        Output of :func:`src.analyzer.find_restriction_sites`; entries with
        exactly one site are drawn as internal radial markers.
    output_path : str, optional
        When given, the figure is written here (PNG, tight bounding box) and
        closed. When omitted the live :class:`~matplotlib.figure.Figure` is
        returned instead, which is what the Streamlit app uses.
    window, step : int
        Sliding-window parameters for the composition panel.
    dpi : int
        Output resolution; 300 by default.

    Returns
    -------
    matplotlib.figure.Figure or str
        The figure, or the path it was written to.
    """
    features = list(features)
    restriction_hits = list(restriction_hits)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.4), dpi=dpi,
                             gridspec_kw={"width_ratios": [1.06, 1.0],
                                          "wspace": 0.22})
    fig.patch.set_facecolor("white")

    _draw_circular_map(axes[0], plasmid_name, seq, features, restriction_hits)
    _draw_composition(axes[1], seq, window, step)

    n_cds = sum(1 for f in features if f.feature_type == "CDS")
    n_unique = sum(1 for h in restriction_hits if h.n_sites == 1)
    n_non = sum(1 for h in restriction_hits if h.n_sites == 0)
    fig.suptitle(
        f"PlasmidQC-Studio  |  {plasmid_name}  |  {len(seq):,} bp  |  "
        f"{len(features)} features ({n_cds} CDS)  |  "
        f"{n_unique} unique cutters, {n_non} non-cutters",
        fontsize=11.5, fontweight="bold", y=0.982)
    fig.subplots_adjust(left=0.015, right=0.955, top=0.905, bottom=0.075,
                        wspace=0.22)

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path
    return fig
