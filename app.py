"""PlasmidQC-Studio interactive web app.

Launch with::

    streamlit run app.py

Upload a circular plasmid FASTA (or use the bundled benchmark plasmid), adjust
the minimum ORF length, and inspect the annotation, restriction/Golden Gate
audit, and nearest-neighbor verification primers alongside a live 300-DPI
dashboard.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

from src.analyzer import (annotate_plasmid, design_amplicon_primers,
                          find_restriction_sites, gc_content, read_fasta,
                          reverse_complement)
from src.visualizer import plot_plasmid_dashboard

DEFAULT_FASTA = os.path.join("data", "pSynBio_sfGFP_01.fasta")
ALT_FASTA = os.path.join("data", "pBioBrick_mCherry_AmpR.fasta")

st.set_page_config(page_title="PlasmidQC-Studio", page_icon="🧬",
                   layout="wide")


def _parse_fasta_text(text: str) -> tuple[str, str]:
    """Parse FASTA text from an upload into ``(name, sequence)``."""
    name, chunks = "", []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name:
                break
            name = line[1:].split()[0] if len(line) > 1 else "uploaded"
            continue
        chunks.append(line)
    if not chunks:
        raise ValueError("No sequence found in the uploaded file.")
    return name or "uploaded", "".join(chunks).upper()


@st.cache_data(show_spinner=False)
def _analyze(seq: str, min_orf: int, flank: int):
    """Cached analysis so slider moves do not recompute everything."""
    features = annotate_plasmid(seq, min_aa=min_orf)
    hits = find_restriction_sites(seq)
    pairs = design_amplicon_primers(seq, features, flank=flank)
    return features, hits, pairs


st.title("🧬 PlasmidQC-Studio")
st.caption("Automated circular plasmid annotation, restriction digest QC, and "
           "nearest-neighbor primer design.")

with st.sidebar:
    st.header("Input")
    source = st.radio("Sequence source",
                      ["Bundled: pSynBio_sfGFP_01",
                       "Bundled: pBioBrick_mCherry_AmpR",
                       "Upload FASTA"], index=0)
    upload = None
    if source == "Upload FASTA":
        upload = st.file_uploader("Plasmid FASTA",
                                  type=["fasta", "fa", "fna", "txt"])

    st.header("Parameters")
    min_orf = st.slider("Minimum ORF length (aa)", min_value=30,
                        max_value=400, value=100, step=10,
                        help="Shorter values expose small ORFs; longer values "
                             "keep only major coding sequences.")
    flank = st.slider("Primer flank (bp)", min_value=20, max_value=150,
                      value=60, step=10,
                      help="Bases of context outside each feature boundary.")
    window = st.slider("GC window (bp)", 40, 400, 120, 20)
    step = st.slider("GC step (bp)", 5, 100, 20, 5)

try:
    if source == "Upload FASTA":
        if upload is None:
            st.info("Upload a FASTA file, or switch to a bundled plasmid in "
                    "the sidebar.")
            st.stop()
        name, seq = _parse_fasta_text(upload.getvalue().decode("utf-8"))
    else:
        path = DEFAULT_FASTA if "pSynBio" in source else ALT_FASTA
        name, seq = read_fasta(path)
except (ValueError, OSError) as exc:
    st.error(f"Could not read the sequence: {exc}")
    st.stop()

bad = set(seq) - set("ACGTN")
if bad:
    st.error(f"Sequence contains unsupported characters: "
             f"{', '.join(sorted(bad))}")
    st.stop()

with st.spinner("Running annotation, digest audit, and primer design..."):
    features, hits, pairs = _analyze(seq, min_orf, flank)

cds = [f for f in features if f.feature_type == "CDS"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Length", f"{len(seq):,} bp")
c2.metric("GC content", f"{gc_content(seq):.2f}%")
c3.metric("Features", len(features), f"{len(cds)} CDS")
c4.metric("Unique cutters", sum(1 for h in hits if h.n_sites == 1))
c5.metric("Primer pairs", len(pairs),
          f"{sum(1 for p in pairs if p.tm_balanced)} Tm-balanced")

st.subheader("Plasmid map and composition profile")
fig = plot_plasmid_dashboard(name, seq, features, hits, window=window,
                             step=step)
st.pyplot(fig, use_container_width=True)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
            facecolor="white")
st.download_button("Download dashboard (300 DPI PNG)", buf.getvalue(),
                   file_name=f"{name}_map_dashboard.png", mime="image/png")

tab_feat, tab_digest, tab_primers, tab_seq = st.tabs(
    ["Annotated features", "Restriction / Golden Gate audit",
     "Verification primers", "Sequence"])

with tab_feat:
    df = pd.DataFrame([f.as_dict() for f in features])
    st.dataframe(df.drop(columns=["color"]), use_container_width=True,
                 hide_index=True)
    st.download_button("Download annotations CSV",
                       df.to_csv(index=False).encode(),
                       file_name=f"{name}_annotations.csv", mime="text/csv")

with tab_digest:
    df = pd.DataFrame([h.as_dict() for h in hits])
    st.dataframe(df, use_container_width=True, hide_index=True)
    gg = [h.enzyme for h in hits
          if h.enzyme_class == "Type IIS" and h.n_sites == 0]
    if gg:
        st.success(f"Golden Gate ready for: {', '.join(gg)} "
                   "(no internal recognition sites).")
    else:
        st.warning("Every Type IIS enzyme has at least one internal site; "
                   "domesticate the backbone before Golden Gate assembly.")
    st.download_button("Download digest CSV",
                       df.to_csv(index=False).encode(),
                       file_name=f"{name}_restriction_digest.csv",
                       mime="text/csv")

with tab_primers:
    rows = [row for pair in pairs for row in pair.as_rows()]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Tm from SantaLucia (1998) unified nearest-neighbor "
               "parameters, 50 mM Na+, 250 nM primer.")
    st.download_button("Download primers CSV",
                       df.to_csv(index=False).encode(),
                       file_name=f"{name}_verification_primers.csv",
                       mime="text/csv")

with tab_seq:
    st.text(f">{name} | {len(seq):,} bp | circular")
    st.code("\n".join(seq[i:i + 70] for i in range(0, len(seq), 70)),
            language=None)
    with st.expander("Reverse complement"):
        rc = reverse_complement(seq)
        st.code("\n".join(rc[i:i + 70] for i in range(0, len(rc), 70)),
                language=None)
