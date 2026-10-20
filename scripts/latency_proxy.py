"""A TCP proxy that delays the first byte of every HTTP response by a fixed latency.

Used in front of a local S3 server to approximate object-store round-trip latency: each
request pays the delay once, independent of response size.
"""

import argparse
import asyncio
import os
import random

COUNTS = {"responses": 0, "bytes": 0}
COUNT_FILE = os.environ.get("PROXY_COUNT_FILE", "results/proxy_counts.txt")


async def pipe_client_to_upstream(reader, writer, state):
    while data := await reader.read(65536):
        state["awaiting"] = True
        writer.write(data)
        await writer.drain()
    writer.close()


async def pipe_upstream_to_client(reader, writer, state, latency, jitter):
    while data := await reader.read(65536):
        if state["awaiting"]:
            state["awaiting"] = False
            COUNTS["responses"] += 1
            await asyncio.sleep((latency + random.uniform(-jitter, jitter)) / 1000)
        COUNTS["bytes"] += len(data)
        with open(COUNT_FILE, "w") as f:
            f.write(f"{COUNTS['responses']} {COUNTS['bytes']}\n")
        writer.write(data)
        await writer.drain()
    writer.close()


async def handle(cr, cw, upstream, latency, jitter):
    host, port = upstream.split(":")
    ur, uw = await asyncio.open_connection(host, int(port))
    state = {"awaiting": False}
    await asyncio.gather(
        pipe_client_to_upstream(cr, uw, state),
        pipe_upstream_to_client(ur, cw, state, latency, jitter),
        return_exceptions=True,
    )


async def main(listen, upstream, latency, jitter):
    host, port = listen.split(":")
    server = await asyncio.start_server(lambda r, w: handle(r, w, upstream, latency, jitter), host, int(port))
    print(f"proxy {listen} -> {upstream} (+{latency}±{jitter} ms per response)", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:19001")
    ap.add_argument("--upstream", default="127.0.0.1:19000")
    ap.add_argument("--latency", type=float, default=20)
    ap.add_argument("--jitter", type=float, default=5)
    a = ap.parse_args()
    asyncio.run(main(a.listen, a.upstream, a.latency, a.jitter))
