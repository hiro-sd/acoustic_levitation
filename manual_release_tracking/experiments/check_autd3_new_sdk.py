"""Read all nine AUTD3 firmware versions without emitting sound."""

import argparse
import sys
from pathlib import Path

from autd3 import FirmwareVersion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.autd_client import AutdController
from tracking.core.autd_sender import make_autd_arrangement


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autd-interface",
        default="",
        help=(
            "AUTD-connected Windows adapter Name from Get-NetAdapter. "
            "Empty uses SDK automatic discovery."
        ),
    )
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Permit connection only for diagnosing an older firmware series.",
    )
    return parser.parse_args()


def version_text(version) -> str:
    if version.is_unknown():
        return "unknown"
    return f"{version.major}.{version.minor}.{version.patch}"


def main() -> None:
    args = parse_args()
    geometry = make_autd_arrangement()

    print(f"Configured devices: {geometry.num_devices()}")
    print(
        "SDK supported firmware series:",
        FirmwareVersion.SUPPORTED_SERIES,
    )

    with AutdController.open(
        geometry,
        interface_name=args.autd_interface,
        require_supported_firmware=not args.allow_unsupported,
    ) as autd:
        print(f"Connected devices: {autd.num_devices()}")
        versions = autd.read_firmware_version()
        for index, version in enumerate(versions):
            print(
                f"Device {index}: "
                f"CPU={version_text(version.cpu)}, "
                f"FPGA={version_text(version.fpga)}, "
                f"supported={version.is_supported()}"
            )


if __name__ == "__main__":
    main()
