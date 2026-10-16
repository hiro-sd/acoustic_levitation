import asyncio

from autd3 import (
    Client,
    ClientConfig,
    FirmwareVersion,
    Interface,
    UdpOption,
)
from autd3.geometry import Autd3, Geometry


def make_geometry() -> Geometry:
    w = Autd3.DEVICE_WIDTH
    h = Autd3.DEVICE_HEIGHT

    positions = [
        (0.0, 0.0),
        (0.0, h),
        (0.0, 2.0 * h),
        (w, 2.0 * h),
        (w, h),
        (w, 0.0),
        (2.0 * w, 0.0),
        (2.0 * w, h),
        (2.0 * w, 2.0 * h),
    ]

    return Geometry(
        [Autd3([x, y, 0.0]) for x, y in positions]
    )


def version_text(version) -> str:
    if version.is_unknown():
        return "unknown"
    return f"{version.major}.{version.minor}.{version.patch}"


async def main() -> None:
    geometry = make_geometry()

    print(f"Configured devices: {geometry.num_devices()}")
    print(
        "SDK supported firmware series:",
        FirmwareVersion.SUPPORTED_SERIES,
    )

    # 通常は自動検出でよい。
    link = UdpOption()

    # 自動検出できない場合は、Get-NetAdapterで調べた名前を指定する。
    # link = UdpOption(iface=Interface.Name("Ethernet"))

    config = ClientConfig(
        require_supported_firmware=True,
    )

    client = await Client.open(geometry, link, config)

    try:
        print(f"Connected devices: {client.num_devices()}")

        versions = await client.read_firmware_version()

        for index, version in enumerate(versions):
            print(
                f"Device {index}: "
                f"CPU={version_text(version.cpu)}, "
                f"FPGA={version_text(version.fpga)}, "
                f"supported={version.is_supported()}"
            )
    finally:
        try:
            await client.silent_stop()
        finally:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())