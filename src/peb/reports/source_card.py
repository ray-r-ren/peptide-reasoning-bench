"""Source-card generation."""

from __future__ import annotations

from peb.schemas import SourceManifestEntry


def render_source_card(entry: SourceManifestEntry) -> str:
    policy = entry.redistribution_policy
    return f"""# {entry.name} Source Card

## Role in PEB
Module: `{entry.module}`

Category: {entry.source_category}

Bucket: {entry.source_bucket.value}

Adapter status: {entry.adapter_status.value}

## Source
URL: {entry.url}

Version policy: {entry.source_version}

Retrieval policy: {entry.retrieval_date_policy}

Citation: {entry.citation}

## Expected Fields
{chr(10).join(f"- `{field}`" for field in entry.expected_fields)}

## Redistribution Policy
{entry.license_or_usage_note}

Raw redistribution: {policy.raw_data_redistribution.value}

Processed label redistribution: {policy.processed_label_redistribution.value}

Commercial use: {policy.commercial_use.value}

Attribution required: {policy.attribution_required}

Share alike obligation: {policy.share_alike_obligation}

Public leaderboard use: {policy.use_in_public_leaderboard.value}

## Comments
{entry.comments}
"""
