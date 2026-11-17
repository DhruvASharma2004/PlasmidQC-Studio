"""Unit tests for the PlasmidQC-Studio analytical core.

Run from the repository root::

    pytest -q
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer import (annotate_plasmid, calculate_tm_nn,
                          design_amplicon_primers, find_orfs,
                          find_restriction_sites, gc_content, read_fasta,
                          reverse_complement, sliding_gc_skew)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")
BENCHMARKS = ["pSynBio_sfGFP_01.fasta", "pBioBrick_mCherry_AmpR.fasta"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=BENCHMARKS)
def benchmark(request):
    """Each bundled benchmark plasmid as ``(name, sequence)``."""
    return read_fasta(os.path.join(DATA_DIR, request.param))


# ---------------------------------------------------------------------------
# Reverse complement
# ---------------------------------------------------------------------------

def test_reverse_complement_basic():
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("") == ""


def test_reverse_complement_is_an_involution():
    seq = "ATGCGTACGTTAGCCATGGATCCGAATTCAAGCTT"
    assert reverse_complement(reverse_complement(seq)) == seq


def test_reverse_complement_handles_case_and_whitespace():
    assert reverse_complement("atgc") == "GCAT"
    assert reverse_complement("ATG\nCGT ") == reverse_complement("ATGCGT")


def test_reverse_complement_palindromic_sites():
    for site in ("GAATTC", "GGATCC", "AAGCTT", "GCGGCCGC", "CTCGAG", "CATATG"):
        assert reverse_complement(site) == site, f"{site} should be palindromic"


def test_reverse_complement_type_iis_sites_are_not_palindromic():
    assert reverse_complement("GGTCTC") == "GAGACC"
    assert reverse_complement("CGTCTC") == "GAGACG"


# ---------------------------------------------------------------------------
# GC content and skew
# ---------------------------------------------------------------------------

def test_gc_content_extremes():
    assert gc_content("GGCC") == 100.0
    assert gc_content("ATAT") == 0.0
    assert gc_content("ATGC") == 50.0
    assert gc_content("") == 0.0


def test_gc_content_is_strand_symmetric():
    seq = "ATGCGTACGTTAGCCATGGATCC"
    assert gc_content(seq) == pytest.approx(gc_content(reverse_complement(seq)))


def test_sliding_gc_skew_shapes_and_wrap_around():
    seq = "ATGC" * 100          # 400 bp
    positions, gc, skew, cumulative = sliding_gc_skew(seq, window=100, step=50)
    assert len(positions) == len(gc) == len(skew) == len(cumulative)
    assert len(positions) == 8          # 400 / 50, circular -> no short tail
    assert all(0.0 <= g <= 100.0 for g in gc)
    assert all(-1.0 <= s <= 1.0 for s in skew)
    # Equal G and C in every window -> zero skew, flat cumulative.
    assert skew == pytest.approx([0.0] * len(skew))
    assert cumulative[-1] == pytest.approx(0.0)


def test_sliding_gc_skew_sign_follows_g_minus_c():
    positions, gc, skew, cumulative = sliding_gc_skew("G" * 200, window=50,
                                                      step=50)
    assert skew == pytest.approx([1.0] * len(skew))
    assert cumulative[-1] == pytest.approx(float(len(skew)))
    assert gc == pytest.approx([100.0] * len(gc))


def test_sliding_gc_skew_cumulative_is_running_sum():
    _, _, skew, cumulative = sliding_gc_skew("ATGGCCAT" * 40, window=60,
                                             step=30)
    running = 0.0
    for s, c in zip(skew, cumulative):
        running += s
        assert c == pytest.approx(running)


def test_sliding_gc_skew_empty_sequence():
    assert sliding_gc_skew("") == ([], [], [], [])


# ---------------------------------------------------------------------------
# Restriction site detection
# ---------------------------------------------------------------------------

def _hits_by_enzyme(seq, **kw):
    return {h.enzyme: h for h in find_restriction_sites(seq, **kw)}


def test_restriction_unique_cutter_detected():
    seq = "A" * 60 + "GAATTC" + "T" * 60
    hit = _hits_by_enzyme(seq)["EcoRI"]
    assert hit.n_sites == 1
    assert hit.verdict == "Unique Cutter"
    assert hit.positions == [61]
    assert hit.enzyme_class == "Type IIP"


def test_restriction_multi_cutter_verdict_and_fragments():
    seq = "A" * 50 + "GAATTC" + "T" * 50 + "GAATTC" + "C" * 50
    hit = _hits_by_enzyme(seq)["EcoRI"]
    assert hit.n_sites == 2
    assert hit.verdict == "Multi-cutter (2x)"
    assert sum(hit.fragment_sizes) == len(seq)     # circular digest closes
    assert len(hit.fragment_sizes) == 2


def test_restriction_non_cutter_verdict():
    seq = "AT" * 100
    hit = _hits_by_enzyme(seq)["NotI"]
    assert hit.n_sites == 0
    assert hit.verdict == "Non-cutter"
    assert hit.fragment_sizes == []


def test_restriction_palindrome_counted_once_per_site():
    """A palindromic site matches on both strands but is one cut."""
    seq = "G" * 30 + "GGATCC" + "A" * 30
    assert _hits_by_enzyme(seq)["BamHI"].n_sites == 1


def test_restriction_type_iis_detected_on_both_strands():
    seq = "A" * 40 + "GGTCTC" + "T" * 40 + "GAGACC" + "C" * 40
    hit = _hits_by_enzyme(seq)["BsaI"]
    assert hit.n_sites == 2
    assert sorted(hit.strands) == ["+", "-"]
    assert hit.golden_gate_ready is False


def test_restriction_golden_gate_flag_true_when_absent():
    seq = "ATGCATGCAT" * 20
    for enzyme in ("BsaI", "BsmBI"):
        hit = _hits_by_enzyme(seq)[enzyme]
        assert hit.n_sites == 0
        assert hit.golden_gate_ready is True


def test_restriction_site_spanning_the_origin_is_found():
    """A site split across position 1 must still be detected when circular."""
    seq = "TTC" + "A" * 80 + "GAA"          # GAA|TTC across the junction
    assert _hits_by_enzyme(seq)["EcoRI"].n_sites == 1
    assert _hits_by_enzyme(seq, circular=False)["EcoRI"].n_sites == 0


def test_benchmark_backbones_are_golden_gate_ready(benchmark):
    _name, seq = benchmark
    hits = _hits_by_enzyme(seq)
    assert hits["BsaI"].n_sites == 0
    assert hits["BsmBI"].n_sites == 0


def test_benchmark_has_unique_cloning_sites(benchmark):
    _name, seq = benchmark
    hits = _hits_by_enzyme(seq)
    unique = [e for e, h in hits.items() if h.verdict == "Unique Cutter"]
    assert "EcoRI" in unique
    assert len(unique) >= 4


# ---------------------------------------------------------------------------
# Nearest-neighbor thermodynamics
# ---------------------------------------------------------------------------

def test_tm_nn_returns_plausible_value_for_a_standard_primer():
    tm = calculate_tm_nn("GTAAAACGACGGCCAGT")      # M13 -21 forward
    assert 45.0 < tm < 60.0


def test_tm_nn_gc_rich_primer_is_hotter_than_at_rich():
    gc_rich = calculate_tm_nn("GCGCGCGCGCGCGCGCGCGC")
    at_rich = calculate_tm_nn("ATATATATATATATATATAT")
    assert gc_rich > at_rich + 25.0


def test_tm_nn_increases_with_length():
    base = "ATGCGTACGTTAGCCATGGA"
    assert calculate_tm_nn(base) < calculate_tm_nn(base + "TCCGAATTCA")


def test_tm_nn_is_strand_symmetric():
    seq = "ATGCGTACGTTAGCCATGGATC"
    assert calculate_tm_nn(seq) == pytest.approx(
        calculate_tm_nn(reverse_complement(seq)), abs=0.05)


def test_tm_nn_salt_and_concentration_dependence():
    seq = "ATGCGTACGTTAGCCATGGATC"
    low_salt = calculate_tm_nn(seq, na_conc_m=0.01)
    high_salt = calculate_tm_nn(seq, na_conc_m=1.0)
    assert high_salt > low_salt
    dilute = calculate_tm_nn(seq, primer_conc_m=25e-9)
    concentrated = calculate_tm_nn(seq, primer_conc_m=2.5e-6)
    assert concentrated > dilute


def test_tm_nn_components_are_exposed():
    tm, dh, ds = calculate_tm_nn("ATGCGTACGTTAGCCATGGATC",
                                 return_components=True)
    assert dh < 0.0            # duplex formation is exothermic
    assert ds < 0.0            # and entropically unfavourable
    assert 40.0 < tm < 75.0


def test_tm_nn_matches_hand_computed_value():
    """Independent re-derivation of the published NN equation for 'ATGC'."""
    import math
    dh = -7.2 + (-8.5) + (-9.8)        # AT + TG + GC
    ds = -20.4 + (-22.7) + (-24.4)
    dh += 2.3 + 0.1                    # terminal A-T, then terminal G-C
    ds += 4.1 + (-2.8)
    ds += 0.368 * 3 * math.log(50e-3)
    expected = (dh * 1000.0) / (ds + 1.987 * math.log(250e-9)) - 273.15
    assert calculate_tm_nn("ATGC") == pytest.approx(expected, abs=1e-6)


def test_tm_nn_rejects_bad_input():
    with pytest.raises(ValueError):
        calculate_tm_nn("A")
    with pytest.raises(ValueError):
        calculate_tm_nn("ATGCN")


# ---------------------------------------------------------------------------
# ORF discovery
# ---------------------------------------------------------------------------

def _make_orf(n_aa: int) -> str:
    """An ORF coding ``n_aa`` residues; the returned DNA is (n_aa + 1) * 3 bp
    because ``length_bp`` includes the terminating stop codon."""
    return "ATG" + "AAA" * (n_aa - 1) + "TAA"


def test_find_orfs_detects_a_plus_strand_cds():
    seq = "GGGG" + _make_orf(120) + "CCCC"
    orfs = find_orfs(seq, min_aa=100)
    assert len(orfs) == 1
    orf = orfs[0]
    assert orf.strand == "+"
    assert orf.feature_type == "CDS"
    assert orf.start == 5
    assert orf.length_bp == 121 * 3        # 120 codons + stop
    assert "120 aa" in orf.notes


def test_find_orfs_detects_a_minus_strand_cds():
    seq = "GGGG" + reverse_complement(_make_orf(130)) + "CCCC"
    orfs = find_orfs(seq, min_aa=100)
    assert len(orfs) == 1
    assert orfs[0].strand == "-"
    assert orfs[0].length_bp == 131 * 3    # 130 codons + stop


def test_find_orfs_respects_min_aa_threshold():
    seq = "GG" + _make_orf(60) + "CC"
    assert find_orfs(seq, min_aa=100) == []
    assert len(find_orfs(seq, min_aa=50)) == 1


def test_find_orfs_filters_nested_shadow_orfs():
    """An internal in-frame ATG must not be reported as a second CDS."""
    body = "ATG" + "AAA" * 40 + "ATG" + "AAA" * 80 + "TAA"
    orfs = find_orfs("CC" + body + "GG", min_aa=100)
    assert len(orfs) == 1
    assert orfs[0].length_bp == len(body)


def test_find_orfs_handles_origin_spanning_cds():
    """A CDS split across position 1 is found once and flagged as wrapping."""
    orf = _make_orf(110)
    rotated = orf[150:] + "TTTT" + orf[:150]      # cut mid-ORF, rotate
    orfs = find_orfs(rotated, min_aa=100, circular=True)
    assert len(orfs) == 1
    assert orfs[0].wraps_origin is True
    assert orfs[0].length_bp == len(orf)
    assert find_orfs(rotated, min_aa=100, circular=False) == []


def test_find_orfs_stops_at_the_first_in_frame_stop():
    seq = "ATG" + "AAA" * 150 + "TAA" + "AAA" * 60 + "TAG"
    orfs = find_orfs(seq, min_aa=100)
    assert orfs[0].length_bp == 3 + 150 * 3 + 3   # ATG + 150 codons + stop


def test_find_orfs_empty_for_short_input():
    assert find_orfs("ATG") == []


def test_benchmark_orfs_are_two_primary_cds(benchmark):
    _name, seq = benchmark
    orfs = find_orfs(seq, min_aa=100)
    assert len(orfs) == 2
    assert [o.name for o in orfs] == ["CDS_1", "CDS_2"]
    for orf in orfs:
        assert orf.length_bp % 3 == 0
        assert orf.length_bp >= 303


# ---------------------------------------------------------------------------
# Annotation and primer design integration
# ---------------------------------------------------------------------------

def test_annotate_plasmid_finds_curated_parts(benchmark):
    _name, seq = benchmark
    features = annotate_plasmid(seq, min_aa=100)
    names = {f.name.split("_")[0] + "_" + f.name.split("_")[1]
             if f.name.count("_") >= 1 else f.name for f in features}
    types = {f.feature_type for f in features}
    assert {"promoter", "RBS", "terminator", "rep_origin", "CDS"} <= types
    assert features == sorted(features, key=lambda f: (f.start, f.name))
    assert all(1 <= f.start <= len(seq) for f in features)
    assert all(1 <= f.end <= len(seq) for f in features)
    assert any("ColE1" in n for n in names)


def test_annotate_plasmid_coordinates_recover_the_motif(benchmark):
    """A feature's reported span must contain the sequence it was called from."""
    _name, seq = benchmark
    doubled = seq + seq
    for feat in annotate_plasmid(seq, min_aa=100):
        sub = doubled[feat.start - 1:feat.start - 1 + feat.length_bp]
        assert len(sub) == feat.length_bp
        if feat.feature_type == "CDS":
            coding = sub if feat.strand == "+" else reverse_complement(sub)
            assert coding.startswith("ATG")
            assert coding[-3:] in ("TAA", "TAG", "TGA")


def test_design_amplicon_primers_brackets_each_target(benchmark):
    _name, seq = benchmark
    features = annotate_plasmid(seq, min_aa=100)
    pairs = design_amplicon_primers(seq, features, flank=60)
    assert pairs, "expected at least one verification amplicon"
    for pair in pairs:
        assert pair.forward.strand == "+"
        assert pair.reverse.strand == "-"
        assert 18 <= pair.forward.length <= 27
        assert 18 <= pair.reverse.length <= 27
        assert pair.amplicon_bp > 100
        assert pair.delta_tm_c == pytest.approx(
            abs(pair.forward.tm_c - pair.reverse.tm_c), abs=0.011)
        assert pair.tm_balanced is (pair.delta_tm_c <= 3.0)


def test_design_amplicon_primers_are_tm_balanced(benchmark):
    _name, seq = benchmark
    pairs = design_amplicon_primers(seq)
    assert all(p.tm_balanced for p in pairs)


def test_primer_sequences_match_the_template(benchmark):
    """Every designed oligo must be an exact substring of one strand."""
    _name, seq = benchmark
    doubled = seq + seq
    for pair in design_amplicon_primers(seq):
        assert pair.forward.sequence in doubled
        assert reverse_complement(pair.reverse.sequence) in doubled


def test_read_fasta_round_trip(benchmark):
    name, seq = benchmark
    assert name
    assert 3100 <= len(seq) <= 3200
    assert set(seq) <= set("ACGT")
