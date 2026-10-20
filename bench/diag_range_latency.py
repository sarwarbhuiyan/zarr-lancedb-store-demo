"""Does the test S3 server slow down ranged reads of large objects? Time 64 KB range GETs directly (no proxy)."""

import statistics
import time

import boto3

s3 = boto3.client("s3", endpoint_url="http://localhost:19000", aws_access_key_id="minioadmin",
                  aws_secret_access_key="minioadmin", region_name="us-east-1")
objs = []
for prefix in ("lance_small.lance/data/", "icechunk_small.icechunk/chunks/", "zarr_sharded.zarr/reflectance/c/"):
    page = s3.list_objects_v2(Bucket="bench", Prefix=prefix, MaxKeys=1000)
    contents = sorted(page.get("Contents", []), key=lambda o: -o["Size"])
    if contents:
        objs.append(contents[0])
for o in objs:
    times = []
    for i in range(10):
        start = (i * 997_331) % max(1, o["Size"] - 65536)
        t0 = time.perf_counter()
        s3.get_object(Bucket="bench", Key=o["Key"], Range=f"bytes={start}-{start + 65535}")["Body"].read()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"{o['Size'] / 1e6:8.1f} MB object {o['Key'][:48]:48} 64 KB range GET median {statistics.median(times):6.1f} ms")
