"""Collect the reads zarr issues in one event-loop tick and resolve them with one blocking call.

zarr-python calls `Store.get` once per chunk, with up to `async.concurrency` calls in flight.
Each store here turns every call made in the same tick into one Lance read (a local `take`,
blob range reads, or one REST take), run in a worker thread so the event loop stays free.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Generic, TypeVar

Req = TypeVar("Req")
Res = TypeVar("Res")


class TickBatcher(Generic[Req, Res]):
    def __init__(self, fetch: Callable[[list[Req]], list[Res]]) -> None:
        """`fetch` takes the batch's requests and returns one result per request, in order."""
        self._fetch = fetch
        self._pending: list[tuple[Req, asyncio.Future]] = []

    async def submit(self, request: Req) -> Res:
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future = loop.create_future()
        if not self._pending:
            loop.call_soon(self._flush)
        self._pending.append((request, waiter))
        return await waiter

    def _flush(self) -> None:
        batch, self._pending = self._pending, []
        fut = asyncio.get_running_loop().run_in_executor(None, self._fetch, [r for r, _ in batch])

        def done(f: asyncio.Future) -> None:
            exc = f.exception()
            for i, (_, waiter) in enumerate(batch):
                if waiter.done():
                    continue
                if exc is not None:
                    waiter.set_exception(exc)
                else:
                    waiter.set_result(f.result()[i])

        fut.add_done_callback(done)
