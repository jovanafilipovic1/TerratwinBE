"""Entry point: ``python -m bandstack``."""

import argparse

from bandstack.build import main

parser = argparse.ArgumentParser(
    prog="bandstack",
    description="Fetch raw stores, align sources onto the target grid, verify, record.",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="re-read every raw layer from its Source even if the stored files cover "
    "the required bounds (verified re-fetch; the fetch gate runs either way)",
)
main(force=parser.parse_args().force)
