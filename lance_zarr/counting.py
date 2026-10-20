"""A pass-through store that counts the requests zarr makes and the bytes it gets back."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass

from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import WrapperStore


@dataclass
class Counts:
    gets: int = 0
    ranged_gets: int = 0
    bytes: int = 0


class CountingStore(WrapperStore):
    def __init__(self, store) -> None:
        super().__init__(store)
        self.counts = Counts()

    def reset(self) -> None:
        self.counts = Counts()

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        buf = await self._store.get(key, prototype, byte_range)
        self.counts.gets += 1
        self.counts.ranged_gets += byte_range is not None
        self.counts.bytes += 0 if buf is None else len(buf)
        return buf

    async def get_partial_values(
        self, prototype: BufferPrototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]
    ) -> list[Buffer | None]:
        key_ranges = list(key_ranges)
        out = await self._store.get_partial_values(prototype, key_ranges)
        self.counts.gets += len(key_ranges)
        self.counts.ranged_gets += sum(r is not None for _, r in key_ranges)
        self.counts.bytes += sum(0 if b is None else len(b) for b in out)
        return out

    async def get_ranges(self, key: str, byte_ranges: Sequence[ByteRequest | None], *, prototype: BufferPrototype,
                         **kwargs) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        # Route through the base implementation so every coalesced read goes through self.get
        # (and is counted) instead of being forwarded to the wrapped store uncounted.
        from zarr.abc.store import Store

        async for group in Store.get_ranges(self, key, byte_ranges, prototype=prototype, **kwargs):
            yield group
