"""Synchronous lifecycle wrapper around the asynchronous AUTD3 SDK client.

The camera/control applications in this repository are intentionally
synchronous and already use dedicated worker threads.  AUTD3 SDK 0.11 uses an
async client, so this module owns one asyncio event loop in a private thread
and exposes the small synchronous surface the applications need.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Sequence
from typing import Any

from autd3 import Client, ClientConfig, Interface, UdpOption
from autd3.geometry import Geometry


class AutdController:
    """Thread-safe synchronous facade for :class:`autd3.Client`."""

    def __init__(
        self,
        geometry: Geometry,
        *,
        interface_name: str = "",
        link: UdpOption | None = None,
        require_supported_firmware: bool = True,
        open_timeout_sec: float = 45.0,
        command_timeout_sec: float = 10.0,
    ) -> None:
        self.geometry = geometry
        self.interface_name = str(interface_name).strip()
        self.link = link
        self.require_supported_firmware = bool(require_supported_firmware)
        self.open_timeout_sec = float(open_timeout_sec)
        self.command_timeout_sec = float(command_timeout_sec)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Client | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._open_error: BaseException | None = None
        self._send_lock = threading.Lock()
        self._closed = True

    @classmethod
    def open(
        cls,
        geometry: Geometry,
        *,
        interface_name: str = "",
        link: UdpOption | None = None,
        require_supported_firmware: bool = True,
        open_timeout_sec: float = 45.0,
        command_timeout_sec: float = 10.0,
    ) -> "AutdController":
        """Create a context-manager compatible controller.

        The actual UDP connection is opened by ``__enter__``.  ``link`` is
        injectable so tests can use ``UdpEmulator.option()`` without changing
        production configuration.
        """

        return cls(
            geometry,
            interface_name=interface_name,
            link=link,
            require_supported_firmware=require_supported_firmware,
            open_timeout_sec=open_timeout_sec,
            command_timeout_sec=command_timeout_sec,
        )

    def _make_link(self) -> UdpOption:
        if self.link is not None:
            return self.link
        if self.interface_name:
            return UdpOption(iface=Interface.Name(self.interface_name))
        return UdpOption()

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def open_client() -> Client:
            return await Client.open(
                self.geometry,
                self._make_link(),
                ClientConfig(
                    require_supported_firmware=self.require_supported_firmware,
                ),
            )

        try:
            self._client = loop.run_until_complete(open_client())
            self._closed = False
        except BaseException as exc:
            self._open_error = exc
        finally:
            self._ready.set()

        if self._client is not None:
            loop.run_forever()

        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        loop.close()

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._thread_main,
            name="autd3-client-loop",
            daemon=True,
        )
        self._thread.start()

        if not self._ready.wait(timeout=self.open_timeout_sec):
            raise TimeoutError(
                "Timed out while opening AUTD3 client "
                f"after {self.open_timeout_sec:.1f} s"
            )
        if self._open_error is not None:
            raise RuntimeError("Failed to open AUTD3 client") from self._open_error
        if self._client is None or self._loop is None:
            raise RuntimeError("AUTD3 client did not initialize")

        print(
            f"[AUTD] Connected devices: {self._client.num_devices()} "
            f"(interface={self.interface_name or 'Auto'})"
        )

    def _run(self, coroutine, *, timeout: float | None = None):
        if self._closed or self._loop is None or self._client is None:
            raise RuntimeError("AUTD3 client is not open")

        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(
                timeout=self.command_timeout_sec if timeout is None else timeout
            )
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("AUTD3 command timed out") from exc

    def send(self, command: Any) -> None:
        async def send_command() -> None:
            assert self._client is not None
            await self._client.send(command)

        with self._send_lock:
            self._run(send_command())

    def silent_stop(self) -> None:
        async def stop_output() -> None:
            assert self._client is not None
            await self._client.silent_stop()

        with self._send_lock:
            self._run(stop_output())

    def read_firmware_version(self) -> Sequence[Any]:
        async def read_versions():
            assert self._client is not None
            return await self._client.read_firmware_version()

        with self._send_lock:
            return self._run(read_versions())

    def num_devices(self) -> int:
        if self._client is None:
            raise RuntimeError("AUTD3 client is not open")
        return int(self._client.num_devices())

    def center(self):
        return self.geometry.center()

    def close(self) -> None:
        if self._thread is None:
            return

        loop = self._loop
        if not self._closed and self._client is not None and loop is not None:
            async def close_client() -> None:
                assert self._client is not None
                try:
                    await self._client.silent_stop()
                finally:
                    await self._client.close()

            try:
                with self._send_lock:
                    self._run(
                        close_client(),
                        timeout=max(self.command_timeout_sec, 15.0),
                    )
            finally:
                self._closed = True
                loop.call_soon_threadsafe(loop.stop)
        elif loop is not None:
            loop.call_soon_threadsafe(loop.stop)

        self._thread.join(timeout=max(self.command_timeout_sec, 15.0))
        self._thread = None
        self._client = None
        self._loop = None

    def __enter__(self) -> "AutdController":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

