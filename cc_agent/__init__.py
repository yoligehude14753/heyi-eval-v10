"""cc_agent — Python-only restricted SHOWCASE runner.

In v9 this package was a Docker image with shell + docker socket. PR#5
strips both and turns it into a regular Python module with the same
trust profile as ``curator/``: no shell, no docker, no host filesystem
access outside the run directory. The only operation it performs is
calling ``heyi_engine.HeyiEngineClient`` (for prompt planning + grading)
and the deployed evaluation engine via the HTTP boundary in
``capability._http_post_chat`` (for actually running the prompts).

The package retains the ``cc_agent`` name for git-history continuity;
the noun "Claude Code" no longer applies in v10.
"""
from __future__ import annotations

from .showcase_runner import (
    ShowcaseResult,
    ShowcaseRunnerError,
    execute_showcase,
)

__all__ = ["ShowcaseResult", "ShowcaseRunnerError", "execute_showcase"]
