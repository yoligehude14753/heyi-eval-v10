"""HuggingFace model card → structured metadata enrichment.

For each discovered candidate, fetch its model card markdown from HF
(via hf-mirror.com), feed it to a local LLM through CCR, and persist a
strictly-schema'd curated.json with fields like publisher, contributors,
claimed_strengths, innovations, interesting_points, etc.

This is the v9 "CURATE" stage of the 9-stage pipeline. Owner of this
output is the orchestrator's CURATE stub today; once T3 lands, the stub
is replaced with a call to curator.enricher.enrich_one().
"""
