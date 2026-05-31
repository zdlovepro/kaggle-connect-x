"""Compatibility entrypoint for submission build pipeline.

This module re-exports the public API from evaluate.build_submission so
training/evaluation scripts can import `build_submission` from project root.
"""

from evaluate.build_submission import (  # noqa: F401
    BUILD_END_MARKER,
    BUILD_START_MARKER,
    BuildConfig,
    DEFAULT_POSITION_HEATMAP,
    DEFAULT_W_HMAP,
    DEFAULT_W_ODD_EVEN,
    DEFAULT_W_SCORE,
    DEFAULT_W_THREAT,
    build_submission,
    build_submission_with_config,
    inject_build_block,
    render_build_block,
    resolve_build_config,
)

