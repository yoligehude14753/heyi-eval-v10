"""HuggingFace model discovery / tracking.

Pulls new model candidates from HF via hf-mirror.com (the only HF endpoint
nv8 can reliably reach inside the corp net) and appends them to an
append-only candidates.jsonl. Two discovery channels:

1. Whitelisted orgs — anything published by canonical foundation-model
   teams (Qwen, deepseek-ai, meta-llama, …). High signal, low noise.
2. Trending — any author, but only if downloads_last_30d ≥ threshold
   AND likes ≥ threshold. Catches breakouts that aren't on our whitelist.

Designed to be run as a systemd timer (e.g. every 4h) — see deploy/.
"""
