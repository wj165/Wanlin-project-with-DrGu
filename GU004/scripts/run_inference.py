"""
CellSAM WSI inference script (research prototype).

See VARIABLE_MAPPING.md for notebook-to-script variable correspondence.

This script defines the minimal, reproducible interface for
CellSAM-based patch-level inference and WSI-level aggregation.

NOTE:
- Core logic is currently implemented and validated in notebooks.
- Implementations will be progressively migrated here.
"""

from typing import Any, Dict


def run_patch_inference(
    image: Any,
    config: Dict[str, Any],
) -> Any:
    """
    Run CellSAM inference on a single image or patch.
    """
    raise NotImplementedError("Implemented in notebooks; to be migrated.")


def aggregate_to_wsi(
    patch_outputs: Any,
    metadata: Dict[str, Any],
) -> Any:
    """
    Aggregate patch-level outputs back to WSI coordinates.
    """
    raise NotImplementedError("Implemented in notebooks; to be migrated.")


def main():
    raise NotImplementedError(
        "CLI entry point will be added after interface stabilization."
    )


if __name__ == "__main__":
    main()
