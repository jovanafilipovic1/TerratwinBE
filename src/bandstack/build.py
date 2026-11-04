"""Build orchestration: ensure raw stores, align sources, verify, record."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from bandstack.grid import from_preset
from bandstack.sources import soilgrids


def _grid_fingerprint(grid) -> dict:
    return {
        "crs": grid.crs.to_string(),
        "transform": [
            grid.transform.a,
            grid.transform.b,
            grid.transform.c,
            grid.transform.d,
            grid.transform.e,
            grid.transform.f,
        ],
        "width": grid.width,
        "height": grid.height,
    }


def main(*, force: bool = False) -> None:
    grid = from_preset("lambert72")
    print(
        f"target grid: {grid.crs.to_string()}, {grid.res:g} m cells, "
        f"{grid.width}×{grid.height} = {grid.width * grid.height:,} cells"
    )

    soilgrids.fetch_raw(grid, force=force)
    path, stats = soilgrids.align(grid)
    checks = soilgrids.verify_aligned(path, stats, grid)

    print("\nper-band summary (inside Study Area mask):")
    for s in stats:
        print(
            f"  {s['layer']:26s} valid {1 - s['nan_inside_frac']:6.1%} "
            f"[{s['min']:8.2f}, {s['max']:8.2f}] {s['unit']}"
        )

    print("\nverification:")
    failed = False
    for c in checks:
        failed |= not c["ok"]
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")

    sidecar = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "grid": _grid_fingerprint(grid),
        "source": "SoilGrids 2.0 (ISRIC) via data/raw/soilgrids (see raw _metadata.json)",
        "resampling": "bilinear (one warp per layer, native IGH -> target grid)",
        "encoding": "float32, decoded physical units, NaN nodata",
        "mask": "Study Area (GISCO NUTS 2021 BE2), all_touched=True, applied at align",
        "bands": stats,
        "verification": checks,
    }
    sidecar_path = soilgrids.ALIGNED_DIR / "_metadata.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\nsidecar written: {sidecar_path}")

    if failed:
        raise SystemExit("verification FAILED")
    print("all checks passed")


if __name__ == "__main__":
    main()
