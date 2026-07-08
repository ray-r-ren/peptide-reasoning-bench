"""Benchmark-card generation."""


def render_benchmark_card(case_counts: dict[str, int], release_status: str) -> str:
    lines = [
        "# Peptide Engineering Benchmark Card",
        "",
        f"Release status: {release_status}",
        "",
        "PEB evaluates evidence-grounded peptide engineering judgment across structure, pose, binding-rank, and human-effect evidence tracks.",
        "",
        "The human-effect track classifies source support, evidence level, direction, claim status, and safety-scope status. It is not an efficacy-prediction task.",
        "",
        "## Case Counts",
    ]
    lines.extend(f"- {track}: {count}" for track, count in sorted(case_counts.items()))
    lines.extend(
        [
            "",
            "## Known Limitations",
            "Source-backed releases may include source-reference-only records and subset metrics depending on coordinate and label availability.",
            "",
            "## Safety Scope",
            "PEB does not validate research peptides for human use and does not make therapeutic, consumer-health, or human-use claims.",
        ]
    )
    return "\n".join(lines) + "\n"
