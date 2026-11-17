#!/usr/bin/env python3
"""PlasmidQC-Studio command-line interface.

Runs the full QC pipeline on a circular plasmid FASTA and writes four
artifacts into the output directory:

===============================  =========================================
``<name>_annotations.csv``       curated parts + primary CDS calls
``<name>_restriction_digest.csv``  Type IIP / Type IIS cutter audit
``<name>_verification_primers.csv``  nearest-neighbor verification primers
``<name>_map_dashboard.png``     300-DPI circular map + composition profile
===============================  =========================================

Examples
--------
    python cli.py --fasta data/pSynBio_sfGFP_01.fasta
    python cli.py --fasta my_plasmid.fasta --outdir results --min-orf 80
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd

from src.analyzer import (annotate_plasmid, design_amplicon_primers,
                          find_restriction_sites, gc_content, read_fasta)
from src.visualizer import plot_plasmid_dashboard

ANNOTATION_COLUMNS = ["name", "feature_type", "start", "end", "strand",
                      "length_bp", "gc_percent", "wraps_origin", "notes",
                      "color"]
DIGEST_COLUMNS = ["enzyme", "site", "enzyme_class", "n_sites", "verdict",
                  "positions", "strands", "cut_positions", "fragment_sizes",
                  "golden_gate_ready"]
PRIMER_COLUMNS = ["target", "role", "name", "sequence", "length", "start",
                  "end", "strand", "gc_percent", "tm_c", "dh_kcal", "ds_cal",
                  "gc_clamp", "three_prime_base", "amplicon_start",
                  "amplicon_end", "amplicon_bp", "amplicon_gc_percent",
                  "delta_tm_c", "tm_balanced", "verdict", "warnings"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plasmidqc",
        description="Annotate a circular plasmid, audit its restriction and "
                    "Golden Gate sites, design nearest-neighbor verification "
                    "primers, and render a 300-DPI QC dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--fasta", required=True,
                        help="Input single-record plasmid FASTA file")
    parser.add_argument("--outdir", default="results",
                        help="Directory for CSV and PNG outputs")
    parser.add_argument("--min-orf", type=int, default=100, dest="min_orf",
                        help="Minimum ORF length in amino acids")
    parser.add_argument("--window", type=int, default=120,
                        help="Sliding window width (bp) for the GC profile")
    parser.add_argument("--step", type=int, default=20,
                        help="Sliding window step (bp) for the GC profile")
    parser.add_argument("--flank", type=int, default=60,
                        help="Flanking bases outside each feature for primers")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip dashboard rendering (CSV exports only)")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.fasta):
        print(f"error: FASTA not found: {args.fasta}", file=sys.stderr)
        return 2
    if args.min_orf < 1:
        print("error: --min-orf must be >= 1", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    record_name, seq = read_fasta(args.fasta)
    stem = os.path.splitext(os.path.basename(args.fasta))[0]

    print(f"[1/4] loaded {record_name}: {len(seq):,} bp, "
          f"GC {gc_content(seq):.2f}%")

    features = annotate_plasmid(seq, min_aa=args.min_orf)
    n_cds = sum(1 for f in features if f.feature_type == "CDS")
    ann_path = os.path.join(args.outdir, f"{stem}_annotations.csv")
    pd.DataFrame([f.as_dict() for f in features]).reindex(
        columns=ANNOTATION_COLUMNS).to_csv(ann_path, index=False)
    print(f"[2/4] annotated {len(features)} features ({n_cds} CDS) "
          f"-> {ann_path}")

    hits = find_restriction_sites(seq)
    digest_path = os.path.join(args.outdir, f"{stem}_restriction_digest.csv")
    pd.DataFrame([h.as_dict() for h in hits]).reindex(
        columns=DIGEST_COLUMNS).to_csv(digest_path, index=False)
    n_unique = sum(1 for h in hits if h.n_sites == 1)
    n_multi = sum(1 for h in hits if h.n_sites > 1)
    n_non = sum(1 for h in hits if h.n_sites == 0)
    print(f"[3/4] digest audit: {n_unique} unique, {n_multi} multi-cutter, "
          f"{n_non} non-cutter -> {digest_path}")

    pairs = design_amplicon_primers(seq, features, flank=args.flank)
    rows = [row for pair in pairs for row in pair.as_rows()]
    primer_path = os.path.join(args.outdir, f"{stem}_verification_primers.csv")
    pd.DataFrame(rows).reindex(columns=PRIMER_COLUMNS).to_csv(primer_path,
                                                              index=False)
    n_pass = sum(1 for p in pairs if p.tm_balanced)
    print(f"[4/4] designed {len(pairs)} primer pairs "
          f"({n_pass} Tm-balanced) -> {primer_path}")

    if not args.no_plot:
        png_path = os.path.join(args.outdir, f"{stem}_map_dashboard.png")
        plot_plasmid_dashboard(record_name, seq, features, hits,
                               output_path=png_path, window=args.window,
                               step=args.step)
        print(f"      dashboard -> {png_path}")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
