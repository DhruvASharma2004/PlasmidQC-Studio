"""Circular plasmid annotation, restriction-digest QC, and nearest-neighbor primer design.

This module is the analytical core of PlasmidQC-Studio. It is deliberately
dependency-light (standard library only) so that it can be imported by the CLI,
the Streamlit app, and the test-suite without pulling in a plotting stack.

Contents
--------
Sequence utilities
    ``reverse_complement``, ``gc_content``, ``sliding_gc_skew``
Open reading frames
    ``find_orfs`` -- 6-frame, circular-aware, shadow-ORF filtered
Feature annotation
    ``annotate_plasmid`` -- curated synthetic-biology part scanner + CDS calls
Restriction / Golden Gate QC
    ``find_restriction_sites`` -- Type IIP and Type IIS cutter audit
Primer design
    ``calculate_tm_nn``, ``design_amplicon_primers`` -- SantaLucia (1998)
    unified nearest-neighbor thermodynamics

All coordinates exposed on public dataclasses are **1-based inclusive**, which
is the convention used by GenBank, SnapGene, and Benchling. Internally the
module works in 0-based half-open coordinates on a doubled sequence so that
features spanning the origin of a circular molecule are handled correctly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "AnnotatedFeature",
    "RestrictionHit",
    "Primer",
    "PrimerPair",
    "reverse_complement",
    "gc_content",
    "sliding_gc_skew",
    "find_orfs",
    "annotate_plasmid",
    "find_restriction_sites",
    "calculate_tm_nn",
    "design_amplicon_primers",
    "read_fasta",
    "PART_MOTIFS",
    "RESTRICTION_ENZYMES",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMPLEMENT = str.maketrans("ACGTNacgtnRYKMSWBDHVrykmswbdhv",
                            "TGCANtgcanYRMKSWVHDByrmkswvhdb")

STOP_CODONS = ("TAA", "TAG", "TGA")
START_CODON = "ATG"

#: Curated synthetic-biology parts. Each entry maps a part name to its
#: consensus sequence, a search regex (IUPAC-expanded), a feature class, and a
#: plotting colour. The consensus is what ``data/`` generators embed, so the
#: scanner and the benchmark data stay in lock-step.
PART_MOTIFS: Dict[str, Dict[str, str]] = {
    "ColE1_Origin": {
        "consensus": "TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTT",
        "regex": (r"TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACC"
                  r"ACCGCTACCAGCGGTGGTTTGTTT"),
        "feature_type": "rep_origin",
        "color": "#6a51a3",
    },
    "T7_Promoter": {
        "consensus": "TAATACGACTCACTATAGGG",
        "regex": r"TAATACGACTCACTATAG",
        "feature_type": "promoter",
        "color": "#238b45",
    },
    "J23100_Promoter": {
        "consensus": "TTGACGGCTAGCTCAGTCCTAGGTACAGTGCTAGC",
        "regex": r"TTGACGGCTAGCTCAGTCCTAGG[TG]ACAGTGCTAGC",
        "feature_type": "promoter",
        "color": "#41ab5d",
    },
    "T7_RBS": {
        "consensus": "AAGGAGATATACAT",
        "regex": r"AAGGAGATATACAT",
        "feature_type": "RBS",
        "color": "#ec7014",
    },
    "Strong_RBS_B0034": {
        "consensus": "AAAGAGGAGAAA",
        "regex": r"AAAGAGGAGAAA",
        "feature_type": "RBS",
        "color": "#fe9929",
    },
    "T7_Terminator": {
        "consensus": "CTAGCATAACCCCTTGGGGCCTCTAAACGGGTCTTGAGGGGTTTTTTG",
        "regex": r"CTAGCATAACCCCTTGGGGCCTCTAAACGGGTCTTGAGGGGTTTTTTG",
        "feature_type": "terminator",
        "color": "#cb181d",
    },
}

#: Type IIP (palindromic) workhorse enzymes plus the two Type IIS enzymes that
#: matter for Golden Gate assembly. ``cut_offset`` is the 5' cut position
#: relative to the start of the recognition site on the top strand.
RESTRICTION_ENZYMES: Dict[str, Dict[str, object]] = {
    "EcoRI":  {"site": "GAATTC",   "class": "Type IIP", "cut_offset": 1,  "overhang": "5'"},
    "BamHI":  {"site": "GGATCC",   "class": "Type IIP", "cut_offset": 1,  "overhang": "5'"},
    "HindIII": {"site": "AAGCTT",  "class": "Type IIP", "cut_offset": 1,  "overhang": "5'"},
    "NotI":   {"site": "GCGGCCGC", "class": "Type IIP", "cut_offset": 2,  "overhang": "5'"},
    "XhoI":   {"site": "CTCGAG",   "class": "Type IIP", "cut_offset": 1,  "overhang": "5'"},
    "NdeI":   {"site": "CATATG",   "class": "Type IIP", "cut_offset": 2,  "overhang": "5'"},
    "BsaI":   {"site": "GGTCTC",   "class": "Type IIS", "cut_offset": 7,  "overhang": "5' (N1, 4-nt)"},
    "BsmBI":  {"site": "CGTCTC",   "class": "Type IIS", "cut_offset": 7,  "overhang": "5' (N1, 4-nt)"},
}

_ORF_COLORS = ["#2171b5", "#08519c", "#4292c6", "#6baed6", "#9ecae1"]

# SantaLucia (1998) unified nearest-neighbor parameters.
# dH in kcal/mol, dS in cal/(mol*K), 1 M NaCl reference state.
NN_DH: Dict[str, float] = {
    "AA": -7.9, "TT": -7.9,
    "AT": -7.2,
    "TA": -7.2,
    "CA": -8.5, "TG": -8.5,
    "GT": -8.4, "AC": -8.4,
    "CT": -7.8, "AG": -7.8,
    "GA": -8.2, "TC": -8.2,
    "CG": -10.6,
    "GC": -9.8,
    "GG": -8.0, "CC": -8.0,
}
NN_DS: Dict[str, float] = {
    "AA": -22.2, "TT": -22.2,
    "AT": -20.4,
    "TA": -21.3,
    "CA": -22.7, "TG": -22.7,
    "GT": -22.4, "AC": -22.4,
    "CT": -21.0, "AG": -21.0,
    "GA": -22.2, "TC": -22.2,
    "CG": -27.2,
    "GC": -24.4,
    "GG": -19.9, "CC": -19.9,
}
#: Helix-initiation penalties, keyed by the identity of the terminal base pair.
NN_INIT_DH = {"GC": 0.1, "AT": 2.3}
NN_INIT_DS = {"GC": -2.8, "AT": 4.1}

R_GAS = 1.987  # cal/(mol*K)

DEFAULT_PRIMER_CONC_M = 250e-9   # 250 nM each primer
DEFAULT_NA_CONC_M = 50e-3        # 50 mM Na+


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedFeature:
    """A single annotated element on the plasmid (1-based inclusive coords)."""

    name: str
    feature_type: str
    start: int
    end: int
    strand: str          # '+' or '-'
    length_bp: int
    gc_percent: float
    color: str = "#808080"
    wraps_origin: bool = False
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RestrictionHit:
    """Per-enzyme summary of a restriction / Golden Gate digest audit."""

    enzyme: str
    site: str
    enzyme_class: str
    n_sites: int
    verdict: str                      # Unique Cutter | Multi-cutter (Nx) | Non-cutter
    positions: List[int] = field(default_factory=list)
    strands: List[str] = field(default_factory=list)
    cut_positions: List[int] = field(default_factory=list)
    fragment_sizes: List[int] = field(default_factory=list)
    golden_gate_ready: Optional[bool] = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = ";".join(str(p) for p in self.positions)
        d["strands"] = ";".join(self.strands)
        d["cut_positions"] = ";".join(str(p) for p in self.cut_positions)
        d["fragment_sizes"] = ";".join(str(p) for p in self.fragment_sizes)
        return d


@dataclass
class Primer:
    """A single oligonucleotide with nearest-neighbor thermodynamics."""

    name: str
    sequence: str
    start: int           # 1-based inclusive, position on the plus strand
    end: int             # 1-based inclusive
    strand: str
    length: int
    gc_percent: float
    tm_c: float
    dh_kcal: float
    ds_cal: float
    gc_clamp: bool
    three_prime_base: str
    warnings: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PrimerPair:
    """A verification amplicon: forward primer, reverse primer, and QC flags."""

    target: str
    forward: Primer
    reverse: Primer
    amplicon_start: int
    amplicon_end: int
    amplicon_bp: int
    amplicon_gc_percent: float
    delta_tm_c: float
    tm_balanced: bool
    verdict: str

    def as_rows(self) -> List[dict]:
        rows = []
        for role, p in (("forward", self.forward), ("reverse", self.reverse)):
            row = p.as_dict()
            row.update(
                target=self.target,
                role=role,
                amplicon_start=self.amplicon_start,
                amplicon_end=self.amplicon_end,
                amplicon_bp=self.amplicon_bp,
                amplicon_gc_percent=self.amplicon_gc_percent,
                delta_tm_c=self.delta_tm_c,
                tm_balanced=self.tm_balanced,
                verdict=self.verdict,
            )
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def _clean(seq: str) -> str:
    """Upper-case a sequence and strip whitespace/newlines."""
    return "".join(seq.split()).upper()


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of ``seq``.

    IUPAC ambiguity codes are complemented correctly; case is preserved for the
    unambiguous bases.

    >>> reverse_complement("ATGC")
    'GCAT'
    """
    return _clean(seq).translate(_COMPLEMENT)[::-1]


def gc_content(seq: str) -> float:
    """Return G+C percentage of ``seq`` (0.0 for an empty sequence)."""
    s = _clean(seq)
    if not s:
        return 0.0
    gc = s.count("G") + s.count("C")
    return 100.0 * gc / len(s)


def sliding_gc_skew(seq: str, window: int = 120, step: int = 20
                    ) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Sliding-window GC% and GC-skew across a *circular* sequence.

    The molecule is treated as circular: windows that run off the 3' end wrap
    back to position 1, so the profile is continuous and the cumulative skew
    returns toward its starting value.

    Parameters
    ----------
    seq : str
        Plasmid sequence.
    window : int
        Window width in bp.
    step : int
        Window step in bp.

    Returns
    -------
    positions : list[int]
        1-based window midpoints.
    gc : list[float]
        Window GC percentage.
    skew : list[float]
        Window GC-skew, ``(G - C) / (G + C)``; 0.0 where G+C is 0.
    cumulative_skew : list[float]
        Running sum of ``skew``, the classic replichore diagnostic.
    """
    s = _clean(seq)
    n = len(s)
    if n == 0:
        return [], [], [], []
    window = max(1, min(window, n))
    step = max(1, step)

    doubled = s + s
    positions: List[int] = []
    gc: List[float] = []
    skew: List[float] = []
    cumulative: List[float] = []
    running = 0.0

    for start in range(0, n, step):
        win = doubled[start:start + window]
        g, c = win.count("G"), win.count("C")
        gc.append(100.0 * (g + c) / len(win))
        sk = (g - c) / (g + c) if (g + c) else 0.0
        skew.append(sk)
        running += sk
        cumulative.append(running)
        positions.append((start + window // 2) % n + 1)

    return positions, gc, skew, cumulative


def read_fasta(path: str) -> Tuple[str, str]:
    """Read a single-record FASTA file.

    Returns ``(record_name, sequence)``. The record name is the first
    whitespace-delimited token of the header; the sequence is upper-cased with
    all whitespace removed. Extra records in the file are ignored with the
    first one winning, which matches how plasmid maps are distributed.
    """
    name, chunks = "", []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:          # second record -- stop
                    break
                name = line[1:].split()[0] if len(line) > 1 else "unnamed"
                continue
            chunks.append(line)
    if not chunks:
        raise ValueError(f"No sequence found in FASTA file: {path}")
    return name or "unnamed", _clean("".join(chunks))


# ---------------------------------------------------------------------------
# ORF finding
# ---------------------------------------------------------------------------

@dataclass
class _RawORF:
    """Internal ORF record in 0-based half-open doubled-sequence coords."""
    u_start: int      # unwrapped start on the plus strand
    u_end: int        # unwrapped end (exclusive) on the plus strand
    strand: str
    frame: int
    length: int
    aa_len: int


def _scan_strand(doubled: str, n: int, min_aa: int, strand: str) -> List[_RawORF]:
    """Scan one strand of a doubled sequence for ATG->stop ORFs."""
    orfs: List[_RawORF] = []
    limit = n                      # starts must lie in the first copy
    max_span = n                   # an ORF cannot be longer than the plasmid
    for i in range(limit):
        if doubled[i:i + 3] != START_CODON:
            continue
        j = i + 3
        stop_at = -1
        hard_stop = min(len(doubled) - 2, i + max_span)
        while j < hard_stop:
            codon = doubled[j:j + 3]
            if codon in STOP_CODONS:
                stop_at = j
                break
            j += 3
        if stop_at < 0:
            continue
        length = stop_at + 3 - i           # includes the stop codon
        aa_len = (stop_at - i) // 3        # residues, stop excluded
        if aa_len < min_aa:
            continue
        orfs.append(_RawORF(u_start=i, u_end=i + length, strand=strand,
                            frame=(i % 3) + 1, length=length, aa_len=aa_len))
    return orfs


def _filter_shadow_orfs(orfs: List[_RawORF], overlap_frac: float = 0.5) -> List[_RawORF]:
    """Drop nested and heavily-overlapping shadow ORFs, longest-first.

    A candidate is discarded when it is fully contained in an already-accepted
    ORF on the same strand, or when it shares more than ``overlap_frac`` of its
    own length with one. This collapses the dozens of in-frame internal ATG
    starts that a naive 6-frame scan reports into one call per real CDS.
    """
    kept: List[_RawORF] = []
    for orf in sorted(orfs, key=lambda o: (-o.length, o.u_start)):
        shadowed = False
        for k in kept:
            if k.strand != orf.strand:
                continue
            lo, hi = max(k.u_start, orf.u_start), min(k.u_end, orf.u_end)
            overlap = max(0, hi - lo)
            if overlap == 0:
                continue
            if orf.u_start >= k.u_start and orf.u_end <= k.u_end:
                shadowed = True
                break
            if overlap / orf.length > overlap_frac:
                shadowed = True
                break
        if not shadowed:
            kept.append(orf)
    return sorted(kept, key=lambda o: o.u_start)


def find_orfs(seq: str, min_aa: int = 100, circular: bool = True
              ) -> List[AnnotatedFeature]:
    """Find primary coding sequences in all six reading frames.

    The scan is circular-aware: the sequence is doubled so that an ORF which
    runs across the numbering origin is found exactly once, and reported with
    ``wraps_origin=True``. After scanning, nested and heavily-overlapping
    shadow ORFs are removed so that each locus yields a single primary CDS
    call (see :func:`_filter_shadow_orfs`).

    Parameters
    ----------
    seq : str
        Plasmid sequence.
    min_aa : int
        Minimum protein length in residues (stop codon excluded).
    circular : bool
        When ``False`` the molecule is treated as linear and ORFs are not
        allowed to wrap.

    Returns
    -------
    list[AnnotatedFeature]
        CDS features named ``CDS_1``, ``CDS_2``, ... in ascending start order,
        with ``notes`` recording the reading frame and protein length.
    """
    s = _clean(seq)
    n = len(s)
    if n < 6:
        return []

    plus_template = (s + s) if circular else s
    minus_source = reverse_complement(s)
    minus_template = (minus_source + minus_source) if circular else minus_source
    scan_n = n if circular else max(1, n - 2)

    raw = _scan_strand(plus_template, scan_n, min_aa, "+")
    raw += _scan_strand(minus_template, scan_n, min_aa, "-")

    # Map minus-strand hits (coords on the reverse complement) back onto the
    # plus strand so that overlap filtering happens in one coordinate system.
    mapped: List[_RawORF] = []
    for orf in raw:
        if orf.strand == "+":
            mapped.append(orf)
        else:
            if circular:
                u_end = 2 * n - orf.u_start            # exclusive
                u_start = 2 * n - orf.u_end
            else:
                u_end = n - orf.u_start
                u_start = n - orf.u_end
            mapped.append(_RawORF(u_start=u_start, u_end=u_end, strand="-",
                                  frame=orf.frame, length=orf.length,
                                  aa_len=orf.aa_len))

    # Collapse duplicate minus-strand calls arising from the doubling.
    seen = set()
    unique: List[_RawORF] = []
    for orf in mapped:
        key = (orf.strand, orf.u_start % n, orf.length)
        if key in seen:
            continue
        seen.add(key)
        unique.append(orf)

    features: List[AnnotatedFeature] = []
    for idx, orf in enumerate(_filter_shadow_orfs(unique), start=1):
        start0 = orf.u_start % n
        end0 = (orf.u_end - 1) % n
        # The span crosses the numbering origin when it runs past the end of
        # the molecule from its normalised start. Testing ``u_end > n`` would
        # be wrong for minus-strand calls, whose unwrapped coordinates are
        # mapped from the reverse complement and can start beyond n.
        wraps = start0 + orf.length > n
        sub = (s + s)[start0:start0 + orf.length]
        if orf.strand == "-":
            sub = reverse_complement(sub)
        features.append(AnnotatedFeature(
            name=f"CDS_{idx}",
            feature_type="CDS",
            start=start0 + 1,
            end=end0 + 1,
            strand=orf.strand,
            length_bp=orf.length,
            gc_percent=round(gc_content(sub), 2),
            color=_ORF_COLORS[(idx - 1) % len(_ORF_COLORS)],
            wraps_origin=wraps,
            notes=f"frame {orf.strand}{orf.frame}; {orf.aa_len} aa",
        ))

    # Re-number after sorting so CDS_1 is always the most 5' call.
    features.sort(key=lambda f: (f.start, f.strand))
    for idx, feat in enumerate(features, start=1):
        feat.name = f"CDS_{idx}"
        feat.color = _ORF_COLORS[(idx - 1) % len(_ORF_COLORS)]
    return features


# ---------------------------------------------------------------------------
# Feature annotation
# ---------------------------------------------------------------------------

def _find_motif(seq: str, pattern: str, n: int) -> List[Tuple[int, int, str]]:
    """Locate a regex motif on both strands of a circular sequence.

    Returns ``(start0, end0_exclusive, strand)`` tuples in unwrapped plus-strand
    coordinates. Overlapping matches are reported.
    """
    doubled = seq + seq
    hits: List[Tuple[int, int, str]] = []

    for m in re.finditer(f"(?=({pattern}))", doubled):
        start = m.start()
        if start >= n:
            continue
        end = start + len(m.group(1))
        if end - start > n:
            continue
        hits.append((start, end, "+"))

    rc_pattern = _revcomp_regex(pattern)
    for m in re.finditer(f"(?=({rc_pattern}))", doubled):
        start = m.start()
        if start >= n:
            continue
        end = start + len(m.group(1))
        if end - start > n:
            continue
        hits.append((start, end, "-"))

    # Deduplicate wrapped double-reports.
    out, seen = [], set()
    for start, end, strand in hits:
        key = (start % n, end - start, strand)
        if key in seen:
            continue
        seen.add(key)
        out.append((start, end, strand))
    return sorted(out)


def _revcomp_regex(pattern: str) -> str:
    """Reverse-complement a simple regex of bases and ``[...]`` character classes.

    Supports exactly the syntax used in :data:`PART_MOTIFS`: literal bases and
    bracketed alternatives. Quantifiers are not supported and would need a full
    parser, which these curated motifs do not require.
    """
    tokens: List[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i] == "[":
            j = pattern.index("]", i)
            tokens.append(pattern[i:j + 1])
            i = j + 1
        else:
            tokens.append(pattern[i])
            i += 1

    out: List[str] = []
    for tok in reversed(tokens):
        if tok.startswith("["):
            inner = tok[1:-1]
            out.append("[" + "".join(b.translate(_COMPLEMENT) for b in inner) + "]")
        else:
            out.append(tok.translate(_COMPLEMENT))
    return "".join(out)


def annotate_plasmid(seq: str, min_aa: int = 100,
                     motifs: Optional[Dict[str, Dict[str, str]]] = None,
                     circular: bool = True) -> List[AnnotatedFeature]:
    """Annotate a circular plasmid with curated parts and primary CDS calls.

    Combines regex scanning for standard synthetic-biology parts (origins,
    promoters, RBSs, terminators -- see :data:`PART_MOTIFS`) with the 6-frame
    ORF finder, and returns one sorted feature list suitable for export or
    plotting.

    Parameters
    ----------
    seq : str
        Plasmid sequence.
    min_aa : int
        Minimum CDS length in residues, forwarded to :func:`find_orfs`.
    motifs : dict, optional
        Override the curated part table.
    circular : bool
        Treat the molecule as circular (default).

    Returns
    -------
    list[AnnotatedFeature]
        Sorted by start coordinate, then by feature name.
    """
    s = _clean(seq)
    n = len(s)
    table = PART_MOTIFS if motifs is None else motifs
    features: List[AnnotatedFeature] = []

    for name, spec in table.items():
        hits = _find_motif(s, spec["regex"], n) if circular else [
            (m.start(), m.end(), "+") for m in re.finditer(spec["regex"], s)
        ]
        multiple = len(hits) > 1
        for idx, (start, end, strand) in enumerate(hits, start=1):
            sub = (s + s)[start:end]
            if strand == "-":
                sub = reverse_complement(sub)
            features.append(AnnotatedFeature(
                name=f"{name}_{idx}" if multiple else name,
                feature_type=spec["feature_type"],
                start=start % n + 1,
                end=(end - 1) % n + 1,
                strand=strand,
                length_bp=end - start,
                gc_percent=round(gc_content(sub), 2),
                color=spec["color"],
                wraps_origin=end > n,
                notes="curated motif match",
            ))

    features.extend(find_orfs(s, min_aa=min_aa, circular=circular))
    features.sort(key=lambda f: (f.start, f.name))
    return features


# ---------------------------------------------------------------------------
# Restriction / Golden Gate audit
# ---------------------------------------------------------------------------

def find_restriction_sites(seq: str,
                           enzymes: Optional[Dict[str, Dict[str, object]]] = None,
                           circular: bool = True) -> List[RestrictionHit]:
    """Audit a plasmid for Type IIP and Type IIS restriction sites.

    Both strands are scanned. Palindromic Type IIP sites are counted once (a
    palindrome matches itself on the reverse strand); non-palindromic Type IIS
    sites such as BsaI ``GGTCTC`` are reported per strand, since orientation
    determines where the enzyme cuts.

    Each enzyme is classified as:

    * ``Unique Cutter`` -- exactly one site, i.e. usable for linearisation or
      directional cloning;
    * ``Multi-cutter (Nx)`` -- more than one site;
    * ``Non-cutter`` -- absent, i.e. safe for a Golden Gate destination vector.

    For enzymes with at least one site, the resulting digest fragment sizes are
    computed around the circle. ``golden_gate_ready`` is set for the Type IIS
    enzymes and is ``True`` when the backbone carries no internal site.

    Returns
    -------
    list[RestrictionHit]
        Ordered by enzyme class, then descending site count, then name.
    """
    s = _clean(seq)
    n = len(s)
    table = RESTRICTION_ENZYMES if enzymes is None else enzymes
    results: List[RestrictionHit] = []

    for name, spec in table.items():
        site = str(spec["site"])
        rc_site = reverse_complement(site)
        palindromic = site == rc_site
        search_space = (s + s[:len(site) - 1]) if circular else s

        found: List[Tuple[int, str]] = []
        for m in re.finditer(f"(?={re.escape(site)})", search_space):
            if m.start() < n:
                found.append((m.start(), "+"))
        if not palindromic:
            for m in re.finditer(f"(?={re.escape(rc_site)})", search_space):
                if m.start() < n:
                    found.append((m.start(), "-"))
        found.sort()

        n_sites = len(found)
        if n_sites == 0:
            verdict = "Non-cutter"
        elif n_sites == 1:
            verdict = "Unique Cutter"
        else:
            verdict = f"Multi-cutter ({n_sites}x)"

        offset = int(spec["cut_offset"])
        cut_positions: List[int] = []
        for start, strand in found:
            if strand == "+":
                cut = (start + offset) % n
            else:
                cut = (start + len(site) - offset) % n
            cut_positions.append(cut + 1)
        cut_positions.sort()

        fragments: List[int] = []
        if circular and len(cut_positions) >= 2:
            for i in range(len(cut_positions)):
                nxt = cut_positions[(i + 1) % len(cut_positions)]
                cur = cut_positions[i]
                fragments.append((nxt - cur) % n or n)
        elif circular and len(cut_positions) == 1:
            fragments = [n]
        elif cut_positions:
            edges = [0] + cut_positions + [n]
            fragments = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]

        results.append(RestrictionHit(
            enzyme=name,
            site=site,
            enzyme_class=str(spec["class"]),
            n_sites=n_sites,
            verdict=verdict,
            positions=[p % n + 1 for p, _ in found],
            strands=[st for _, st in found],
            cut_positions=cut_positions,
            fragment_sizes=sorted(fragments, reverse=True),
            golden_gate_ready=(n_sites == 0) if spec["class"] == "Type IIS" else None,
        ))

    results.sort(key=lambda r: (r.enzyme_class, -r.n_sites, r.enzyme))
    return results


# ---------------------------------------------------------------------------
# Nearest-neighbor thermodynamics
# ---------------------------------------------------------------------------

def calculate_tm_nn(seq: str,
                    primer_conc_m: float = DEFAULT_PRIMER_CONC_M,
                    na_conc_m: float = DEFAULT_NA_CONC_M,
                    return_components: bool = False):
    """Melting temperature from SantaLucia (1998) unified nearest-neighbor data.

    The duplex enthalpy and entropy are summed over the ten unique Watson-Crick
    nearest-neighbor dimers, helix-initiation penalties are applied at both
    termini according to the identity of the terminal base pair, and the
    entropy is salt-corrected before solving the two-state melting equation:

    .. math::

        \\Delta S^{\\circ}_{[\\mathrm{Na^+}]} = \\Delta S^{\\circ}_{1\\,\\mathrm{M}}
            + 0.368 \\,(N - 1)\\, \\ln[\\mathrm{Na^+}]

        T_m = \\frac{\\Delta H^{\\circ} \\times 1000}
                    {\\Delta S^{\\circ} + R \\ln C_T} - 273.15

    with :math:`R = 1.987\\ \\mathrm{cal\\,mol^{-1}K^{-1}}` and :math:`C_T` the
    total strand concentration.

    Parameters
    ----------
    seq : str
        Primer sequence (>= 2 nt, unambiguous bases).
    primer_conc_m : float
        Total strand concentration in mol/L (default 250 nM).
    na_conc_m : float
        Monovalent salt concentration in mol/L (default 50 mM).
    return_components : bool
        When ``True`` return ``(tm_c, dh_kcal, ds_cal)`` instead of just ``tm_c``.

    Returns
    -------
    float or tuple
        Melting temperature in degrees Celsius, optionally with the summed
        enthalpy (kcal/mol) and salt-corrected entropy (cal/mol/K).

    Raises
    ------
    ValueError
        If the sequence is shorter than 2 nt or contains a non-ACGT base.
    """
    s = _clean(seq)
    if len(s) < 2:
        raise ValueError("Nearest-neighbor Tm requires at least 2 nucleotides")
    if set(s) - set("ACGT"):
        raise ValueError(f"Non-ACGT base in primer sequence: {s}")

    dh = 0.0
    ds = 0.0
    for i in range(len(s) - 1):
        dimer = s[i:i + 2]
        dh += NN_DH[dimer]
        ds += NN_DS[dimer]

    for terminal in (s[0], s[-1]):
        key = "GC" if terminal in "GC" else "AT"
        dh += NN_INIT_DH[key]
        ds += NN_INIT_DS[key]

    ds_salt = ds + 0.368 * (len(s) - 1) * math.log(na_conc_m)
    tm_k = (dh * 1000.0) / (ds_salt + R_GAS * math.log(primer_conc_m))
    tm_c = tm_k - 273.15

    if return_components:
        return tm_c, dh, ds_salt
    return tm_c


def _has_gc_clamp(seq: str) -> bool:
    """True when the 3'-terminal base is G or C (anchors polymerase extension)."""
    return bool(seq) and seq[-1] in "GC"


def _primer_warnings(seq: str, tm_c: float) -> str:
    """Collect standard oligo QC flags into a semicolon-delimited string."""
    flags: List[str] = []
    gc = gc_content(seq)
    if not 40.0 <= gc <= 60.0:
        flags.append(f"GC {gc:.1f}% outside 40-60%")
    if not 55.0 <= tm_c <= 68.0:
        flags.append(f"Tm {tm_c:.1f}C outside 55-68C")
    if not _has_gc_clamp(seq):
        flags.append("no 3' GC clamp")
    for base in "ACGT":
        if base * 5 in seq:
            flags.append(f"{base}-homopolymer >=5")
            break
    if seq[-5:].count("G") + seq[-5:].count("C") >= 5:
        flags.append("3' end GC-rich (mispriming risk)")
    return "; ".join(flags)


def _slice_circular(seq: str, start0: int, length: int) -> str:
    """Take ``length`` bases from ``start0``, wrapping around the origin."""
    n = len(seq)
    start0 %= n
    return (seq + seq)[start0:start0 + length]


def _pick_primer(seq: str, anchor0: int, direction: str,
                 min_len: int = 18, max_len: int = 27,
                 target_tm: float = 60.0,
                 primer_conc_m: float = DEFAULT_PRIMER_CONC_M,
                 na_conc_m: float = DEFAULT_NA_CONC_M
                 ) -> Tuple[str, int, float, float, float]:
    """Walk primer lengths from ``anchor0`` and keep the best-scoring oligo.

    ``direction='fwd'`` builds a plus-strand primer starting at ``anchor0``;
    ``direction='rev'`` builds a reverse primer whose 5' end sits at
    ``anchor0`` on the minus strand. Candidates are scored on |Tm - target|
    with a small bonus for a 3' GC clamp and in-range GC content.
    """
    n = len(seq)
    best = None
    for length in range(min_len, max_len + 1):
        if direction == "fwd":
            start0 = anchor0
            sub = _slice_circular(seq, start0, length)
        else:
            start0 = (anchor0 - length + 1) % n
            sub = reverse_complement(_slice_circular(seq, start0, length))
        if set(sub) - set("ACGT"):
            continue
        tm, dh, ds = calculate_tm_nn(sub, primer_conc_m, na_conc_m,
                                     return_components=True)
        score = abs(tm - target_tm)
        if _has_gc_clamp(sub):
            score -= 1.5
        if 40.0 <= gc_content(sub) <= 60.0:
            score -= 1.0
        if best is None or score < best[0]:
            best = (score, sub, start0, tm, dh, ds)
    if best is None:
        raise ValueError("Could not build a primer with unambiguous bases")
    _, sub, start0, tm, dh, ds = best
    return sub, start0, tm, dh, ds


def design_amplicon_primers(seq: str,
                            features: Optional[Iterable[AnnotatedFeature]] = None,
                            flank: int = 60,
                            target_tm: float = 60.0,
                            max_delta_tm: float = 3.0,
                            primer_conc_m: float = DEFAULT_PRIMER_CONC_M,
                            na_conc_m: float = DEFAULT_NA_CONC_M,
                            targets: Sequence[str] = ("CDS", "promoter",
                                                      "rep_origin"),
                            ) -> List[PrimerPair]:
    """Design colony-PCR / sequencing verification primers around features.

    For every selected feature a forward primer is placed ``flank`` bp upstream
    and a reverse primer ``flank`` bp downstream, so the amplicon brackets the
    whole element and Sanger reads cover both junctions. Primer length is
    optimised between 18 and 27 nt against ``target_tm`` using
    :func:`calculate_tm_nn`, with a 3' GC-clamp preference.

    The pair is flagged ``tm_balanced`` when ``|dTm| <= max_delta_tm``, the
    standard requirement for a single annealing temperature.

    Parameters
    ----------
    seq : str
        Plasmid sequence (treated as circular).
    features : iterable of AnnotatedFeature, optional
        Features to target; annotated afresh when omitted.
    flank : int
        Bases of context outside each feature boundary.
    target_tm, max_delta_tm : float
        Design target and pairing tolerance in degrees Celsius.
    targets : sequence of str
        Feature types to design against.

    Returns
    -------
    list[PrimerPair]
    """
    s = _clean(seq)
    n = len(s)
    if features is None:
        features = annotate_plasmid(s)
    pairs: List[PrimerPair] = []

    for feat in features:
        if feat.feature_type not in targets:
            continue
        fwd_anchor = (feat.start - 1 - flank) % n
        rev_anchor = (feat.end - 1 + flank) % n

        # Two-pass balancing: design each primer against the global target,
        # then re-design each one against its partner's realised Tm. AT-rich
        # flanks cannot always reach 60 C, so matching the pair to each other
        # matters more for a single annealing temperature than matching both
        # to an unreachable ideal.
        fwd_seq, fwd_start0, fwd_tm, fwd_dh, fwd_ds = _pick_primer(
            s, fwd_anchor, "fwd", target_tm=target_tm,
            primer_conc_m=primer_conc_m, na_conc_m=na_conc_m)
        rev_seq, rev_start0, rev_tm, rev_dh, rev_ds = _pick_primer(
            s, rev_anchor, "rev", target_tm=target_tm,
            primer_conc_m=primer_conc_m, na_conc_m=na_conc_m)

        for _ in range(2):
            if abs(fwd_tm - rev_tm) <= 0.5:
                break
            rev_seq, rev_start0, rev_tm, rev_dh, rev_ds = _pick_primer(
                s, rev_anchor, "rev", target_tm=fwd_tm,
                primer_conc_m=primer_conc_m, na_conc_m=na_conc_m)
            fwd_seq, fwd_start0, fwd_tm, fwd_dh, fwd_ds = _pick_primer(
                s, fwd_anchor, "fwd", target_tm=rev_tm,
                primer_conc_m=primer_conc_m, na_conc_m=na_conc_m)

        forward = Primer(
            name=f"{feat.name}_F", sequence=fwd_seq,
            start=fwd_start0 + 1,
            end=(fwd_start0 + len(fwd_seq) - 1) % n + 1,
            strand="+", length=len(fwd_seq),
            gc_percent=round(gc_content(fwd_seq), 2),
            tm_c=round(fwd_tm, 2), dh_kcal=round(fwd_dh, 2),
            ds_cal=round(fwd_ds, 2), gc_clamp=_has_gc_clamp(fwd_seq),
            three_prime_base=fwd_seq[-1],
            warnings=_primer_warnings(fwd_seq, fwd_tm),
        )
        reverse = Primer(
            name=f"{feat.name}_R", sequence=rev_seq,
            start=rev_start0 + 1,
            end=(rev_start0 + len(rev_seq) - 1) % n + 1,
            strand="-", length=len(rev_seq),
            gc_percent=round(gc_content(rev_seq), 2),
            tm_c=round(rev_tm, 2), dh_kcal=round(rev_dh, 2),
            ds_cal=round(rev_ds, 2), gc_clamp=_has_gc_clamp(rev_seq),
            three_prime_base=rev_seq[-1],
            warnings=_primer_warnings(rev_seq, rev_tm),
        )

        amp_start0 = fwd_start0
        amp_end0 = (rev_start0 + len(rev_seq) - 1) % n
        amp_len = (amp_end0 - amp_start0) % n + 1
        amp_seq = _slice_circular(s, amp_start0, amp_len)
        delta_tm = abs(forward.tm_c - reverse.tm_c)
        balanced = delta_tm <= max_delta_tm

        problems = [p for p in (forward.warnings, reverse.warnings) if p]
        if balanced and not problems:
            verdict = "PASS"
        elif balanced:
            verdict = "PASS (with flags)"
        else:
            verdict = "REVIEW (dTm)"

        pairs.append(PrimerPair(
            target=feat.name, forward=forward, reverse=reverse,
            amplicon_start=amp_start0 + 1, amplicon_end=amp_end0 + 1,
            amplicon_bp=amp_len,
            amplicon_gc_percent=round(gc_content(amp_seq), 2),
            delta_tm_c=round(delta_tm, 2), tm_balanced=balanced,
            verdict=verdict,
        ))

    return pairs
