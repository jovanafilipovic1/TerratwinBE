"""Build orchestration: ensure raw stores, align sources, verify, record."""

from __future__ import annotations

from bandstack.grid import from_preset
from bandstack.sources import soilgrids


def main(*, force: bool = False) -> None:
    grid = from_preset("lambert72")
    print(
        f"target grid: {grid.crs.to_string()}, {grid.res:g} m cells, "
        f"{grid.width}×{grid.height} = {grid.width * grid.height:,} cells"
    )

    soilgrids.fetch_raw(grid, force=force)
    checks = soilgrids.ensure_aligned(grid)

    print("\nverification:")
    failed = False
    for c in checks:
        failed |= not c["ok"]
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")

    if failed:
        raise SystemExit("verification FAILED")
    print("all checks passed")
