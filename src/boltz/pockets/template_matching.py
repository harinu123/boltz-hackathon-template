"""Sequence-template matching utilities for allosteric pocket discovery."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}
BACKGROUND_FREQ = {
    "A": 0.074,
    "R": 0.052,
    "N": 0.045,
    "D": 0.054,
    "C": 0.025,
    "Q": 0.034,
    "E": 0.054,
    "G": 0.074,
    "H": 0.026,
    "I": 0.068,
    "L": 0.099,
    "K": 0.058,
    "M": 0.025,
    "F": 0.047,
    "P": 0.039,
    "S": 0.057,
    "T": 0.051,
    "W": 0.013,
    "Y": 0.032,
    "V": 0.073,
}
EPS = 1e-6

_BLOSUM62_RAW = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def _build_blosum62() -> dict[str, dict[str, int]]:
    rows = [row.strip() for row in _BLOSUM62_RAW.strip().splitlines() if row.strip()]
    header = rows[0].split()
    matrix: dict[str, dict[str, int]] = {}
    for row in rows[1:]:
        parts = row.split()
        aa = parts[0]
        scores = [int(x) for x in parts[1:]]
        matrix[aa] = {header[i]: scores[i] for i in range(len(header))}
    return matrix


BLOSUM62 = _build_blosum62()

_FALLBACK_TEMPLATE_LIBRARY = [
    {
        "name": "Kinase_TypeII_DFG_Allosteric",
        "family": "Protein kinase",
        "sequence": "DFGWSLGLAELIHSLN",
        "pocket_indices": [0, 1, 2, 5, 8, 11, 14],
        "description": "Back pocket spanning the DFG motif and alphaC helix used by type-II kinase inhibitors.",
        "source": "PDB:4WAF chain A",
        "min_coverage": 0.35,
    },
    {
        "name": "GPCR_TM6_TM7_NAM",
        "family": "Class A GPCR",
        "sequence": "CWLPFFLMLGLYYMFVV",
        "pocket_indices": [0, 3, 6, 9, 12, 15],
        "description": "Transmembrane 6/7 cleft targeted by negative allosteric modulators of GPCRs.",
        "source": "PDB:6PS6 chain A",
        "min_coverage": 0.30,
    },
    {
        "name": "Nuclear_Receptor_AF2",
        "family": "Nuclear receptor",
        "sequence": "WQRFYQLLLQLLQEQ",
        "pocket_indices": [0, 3, 5, 8, 11, 13],
        "description": "Allosteric AF-2 co-regulator pocket within ligand binding domain.",
        "source": "PDB:5JFA chain A",
        "min_coverage": 0.25,
    },
    {
        "name": "Serine_Protease_Exosite",
        "family": "Serine protease",
        "sequence": "GGAGGNVGDSGGPLV",
        "pocket_indices": [1, 4, 6, 8, 9, 12, 14],
        "description": "Exosite pocket near the catalytic triad engaged by allosteric thrombin modulators.",
        "source": "PDB:6CSE chain A",
        "min_coverage": 0.28,
    },
]


@dataclass(frozen=True)
class Template:
    """Allosteric template definition."""

    name: str
    sequence: str
    pocket_indices: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    min_coverage: float = 0.3

    def __post_init__(self) -> None:
        if not self.sequence:
            raise ValueError("Template sequence must not be empty")
        if any(res not in AA_TO_INDEX for res in self.sequence):
            raise ValueError(f"Template {self.name} contains non-standard residues")
        if any(idx < 0 or idx >= len(self.sequence) for idx in self.pocket_indices):
            raise ValueError(
                f"Pocket indices for template {self.name} must be within the sequence range"
            )
        if self.min_coverage < 0 or self.min_coverage > 1:
            raise ValueError("min_coverage must be between 0 and 1")


@dataclass(slots=True)
class TemplateMatch:
    """Score for aligning an allosteric template to a target sequence."""

    template_name: str
    residues: list[int]
    score: float
    raw_score: float
    z_score: float
    coverage: float
    conservation: float
    confidence: float
    start: int
    end: int
    alignment_depth: int
    window_sequence: str
    consensus_sequence: str
    metadata: dict[str, Any]


def _sanitize_a3m_sequence(seq: str) -> str:
    return "".join(ch for ch in seq.strip() if ch == "-" or ch.isupper())


def _read_a3m(msa_path: Path) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    with msa_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            current.append(line)
        if current:
            sequences.append("".join(current))
    return [_sanitize_a3m_sequence(seq) for seq in sequences]


def _prepare_alignment(
    sequence: str, msa_sequences: Iterable[str]
) -> list[str]:
    target_len = len(sequence)
    alignment: list[str] = []
    for seq in msa_sequences:
        sanitized = _sanitize_a3m_sequence(seq)
        if len(sanitized) == target_len:
            alignment.append(sanitized)
    if not alignment:
        alignment.append(sequence)
    return alignment


def _compute_profile(
    alignment: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    length = len(alignment[0])
    counts = np.zeros((length, len(AA_ALPHABET)), dtype=np.float64)
    coverage = np.zeros(length, dtype=np.float64)
    for seq in alignment:
        for idx, aa in enumerate(seq):
            if aa == "-":
                continue
            if aa not in AA_TO_INDEX:
                continue
            counts[idx, AA_TO_INDEX[aa]] += 1.0
            coverage[idx] += 1.0
    depth = float(len(alignment))
    if depth > 0:
        coverage /= depth
    background = np.array([BACKGROUND_FREQ[aa] for aa in AA_ALPHABET], dtype=np.float64)
    profile = counts + background  # implicit Dirichlet pseudo-count
    row_sums = profile.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    profile /= row_sums
    entropy = -np.sum(profile * np.log(profile + EPS), axis=1)
    max_entropy = math.log(len(AA_ALPHABET))
    conservation = 1.0 - (entropy / max_entropy)
    consensus_indices = profile.argmax(axis=1)
    consensus = [AA_ALPHABET[idx] for idx in consensus_indices]
    return profile, coverage, conservation, consensus


def _score_window(
    template_seq: str,
    profile: np.ndarray,
    coverage: np.ndarray,
    conservation: np.ndarray,
    consensus: Sequence[str],
    start: int,
) -> tuple[float, float, float, float]:
    total = 0.0
    cov_acc = 0.0
    cons_acc = 0.0
    blosum_acc = 0.0
    for offset, tpl_res in enumerate(template_seq):
        pos = start + offset
        freq_vec = profile[pos]
        cov_acc += coverage[pos]
        cons_acc += conservation[pos]
        if tpl_res in AA_TO_INDEX:
            tpl_idx = AA_TO_INDEX[tpl_res]
            prob = freq_vec[tpl_idx]
            total += math.log(prob + EPS) - math.log(BACKGROUND_FREQ[tpl_res])
        consensus_res = consensus[pos]
        blosum_acc += BLOSUM62[tpl_res][consensus_res]
    window_len = len(template_seq)
    cov_mean = cov_acc / window_len if window_len else 0.0
    cons_mean = cons_acc / window_len if window_len else 0.0
    blosum_mean = blosum_acc / window_len if window_len else 0.0
    raw_score = total + 0.2 * blosum_mean
    return raw_score, cov_mean, cons_mean, blosum_mean


class AllostericTemplateMatcher:
    """Template-based pocket detector using MSA-derived sequence profiles."""

    def __init__(self, template_path: Optional[str | Path] = None) -> None:
        self.templates = self._load_templates(template_path)

    @staticmethod
    def _default_template_path() -> Path:
        return Path(__file__).resolve().parents[3] / "hackathon" / "templates" / "allosteric_templates.json"

    @classmethod
    def _load_templates(cls, template_path: Optional[str | Path]) -> list[Template]:
        template_records: list[dict[str, Any]]
        path = Path(template_path) if template_path else cls._default_template_path()
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                template_records = json.load(handle)
        else:
            template_records = _FALLBACK_TEMPLATE_LIBRARY
        templates: list[Template] = []
        for record in template_records:
            known_keys = {"name", "sequence", "pocket_indices", "min_coverage"}
            metadata = {key: value for key, value in record.items() if key not in known_keys}
            templates.append(
                Template(
                    name=record["name"],
                    sequence=record["sequence"],
                    pocket_indices=tuple(record["pocket_indices"]),
                    metadata=metadata,
                    min_coverage=float(record.get("min_coverage", 0.3)),
                )
            )
        return templates

    @staticmethod
    def expand_residue_indices(
        residues: Sequence[int],
        sequence_length: int,
        flank: int = 1,
        max_residues: int = 18,
    ) -> list[int]:
        if not residues:
            return []
        expanded: set[int] = set()
        for residue in residues:
            for offset in range(-flank, flank + 1):
                candidate = residue + offset
                if 1 <= candidate <= sequence_length:
                    expanded.add(candidate)
        if len(expanded) > max_residues:
            centers = np.array(residues, dtype=float)
            sorted_candidates = sorted(
                expanded,
                key=lambda idx: min(abs(idx - center) for center in centers),
            )
            expanded = set(sorted_candidates[:max_residues])
        return sorted(expanded)

    def find_best_matches(
        self,
        sequence: str,
        msa_path: Optional[str | Path],
        top_n: int = 3,
    ) -> list[TemplateMatch]:
        if not sequence:
            return []
        sequence = sequence.strip().upper()
        msa_sequences: Iterable[str] = []
        if msa_path:
            msa_file = Path(msa_path)
            if msa_file.exists():
                msa_sequences = _read_a3m(msa_file)
        alignment = _prepare_alignment(sequence, msa_sequences)
        profile, coverage, conservation, consensus = _compute_profile(alignment)
        seq_len = len(sequence)
        matches: list[TemplateMatch] = []
        for template in self.templates:
            window = len(template.sequence)
            if seq_len < window:
                continue
            window_scores: list[tuple[float, float, float, float, int]] = []
            for start in range(seq_len - window + 1):
                raw_score, cov_mean, cons_mean, blosum_mean = _score_window(
                    template.sequence,
                    profile,
                    coverage,
                    conservation,
                    consensus,
                    start,
                )
                window_scores.append((raw_score, cov_mean, cons_mean, blosum_mean, start))
            if not window_scores:
                continue
            raw_vals = np.array([entry[0] for entry in window_scores], dtype=np.float64)
            composite_vals = np.array(
                [
                    entry[0]
                    + 1.4 * entry[1]
                    + 1.2 * entry[2]
                    + 0.1 * entry[3]
                    for entry in window_scores
                ],
                dtype=np.float64,
            )
            best_idx = int(np.argmax(composite_vals))
            raw_score, cov_mean, cons_mean, blosum_mean, start = window_scores[best_idx]
            if cov_mean < template.min_coverage:
                continue
            residues = [
                start + rel_idx + 1
                for rel_idx in template.pocket_indices
                if 0 <= start + rel_idx < seq_len
            ]
            if not residues:
                continue
            z_score = float((raw_score - raw_vals.mean()) / (raw_vals.std() + EPS))
            confidence = max(
                0.0,
                min(
                    1.0,
                    0.45 * cov_mean
                    + 0.35 * cons_mean
                    + 0.2 * max(0.0, math.tanh(z_score / 2.5)),
                ),
            )
            metadata = dict(template.metadata)
            metadata.update(
                {
                    "min_coverage": template.min_coverage,
                    "blosum_mean": blosum_mean,
                }
            )
            window_seq = sequence[start : start + window]
            consensus_seq = "".join(consensus[start : start + window])
            matches.append(
                TemplateMatch(
                    template_name=template.name,
                    residues=residues,
                    score=composite_vals[best_idx],
                    raw_score=raw_score,
                    z_score=z_score,
                    coverage=cov_mean,
                    conservation=cons_mean,
                    confidence=confidence,
                    start=start,
                    end=start + window,
                    alignment_depth=len(alignment),
                    window_sequence=window_seq,
                    consensus_sequence=consensus_seq,
                    metadata=metadata,
                )
            )
        matches.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        return matches[:top_n]

