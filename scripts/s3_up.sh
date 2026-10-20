#!/usr/bin/env bash
# Local S3 for the benchmark: MinIO on :19000, and toxiproxy on :19001 adding latency to every response.
set -euo pipefail
LATENCY_MS="${LATENCY_MS:-20}"
JITTER_MS="${JITTER_MS:-5}"
docker network inspect zlbench >/dev/null 2>&1 || docker network create zlbench >/dev/null
docker rm -f zlbench-minio zlbench-toxiproxy >/dev/null 2>&1 || true
# create + start rather than run -d: on this Docker Desktop `run -d` with published ports
# has hung after creating the container, while a separate start succeeds.
docker create --name zlbench-minio --network zlbench -p 19000:9000 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  cgr.dev/chainguard/minio:latest server /data >/dev/null
docker create --name zlbench-toxiproxy --network zlbench -p 18474:8474 -p 19001:19001 \
  ghcr.io/shopify/toxiproxy:2.9.0 >/dev/null
docker start zlbench-minio zlbench-toxiproxy >/dev/null
for i in $(seq 1 30); do curl -sf localhost:19000/minio/health/live >/dev/null && curl -sf localhost:18474/version >/dev/null && break; sleep 1; done
uv run python -c "import boto3; boto3.client('s3', endpoint_url='http://localhost:19000', aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin', region_name='us-east-1').create_bucket(Bucket='bench'); print('bucket ready')"
curl -sf -X POST localhost:18474/proxies -d '{"name":"s3","listen":"0.0.0.0:19001","upstream":"zlbench-minio:9000","enabled":true}' >/dev/null
curl -sf -X POST localhost:18474/proxies/s3/toxics \
  -d "{\"name\":\"latency\",\"type\":\"latency\",\"stream\":\"downstream\",\"attributes\":{\"latency\":${LATENCY_MS},\"jitter\":${JITTER_MS}}}" >/dev/null
echo "minio :19000 (direct), toxiproxy :19001 (+${LATENCY_MS}ms ±${JITTER_MS}ms per response)"
