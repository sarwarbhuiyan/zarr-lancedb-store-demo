"""Aggregate GET throughput of the local S3 rig, direct and through the latency proxy.

Docker Desktop's port forwarding caps bandwidth well below real S3, which penalizes layouts
that read more bytes. Run this next to any object-store result to see the ceiling.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

KEY = "zarr_big.zarr/reflectance/c/0/0/0/0"


def client(endpoint):
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id="minioadmin", aws_secret_access_key="minioadmin",
                        region_name="us-east-1", config=Config(max_pool_connections=128))


for endpoint in sys.argv[1:] or ["http://localhost:19000"]:
    c = client(endpoint)
    size = c.head_object(Bucket="bench", Key=KEY)["ContentLength"]
    for par in (1, 8, 32):
        n = par * 4
        t0 = time.perf_counter()
        with ThreadPoolExecutor(par) as ex:
            list(ex.map(lambda _: c.get_object(Bucket="bench", Key=KEY)["Body"].read(), range(n)))
        dt = time.perf_counter() - t0
        print(f"{endpoint} {par:2} parallel x {size / 1e6:.1f} MB: {n * size / dt / 1e6:7.0f} MB/s aggregate")
