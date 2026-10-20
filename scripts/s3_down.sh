#!/usr/bin/env bash
docker rm -f zlbench-minio zlbench-toxiproxy >/dev/null 2>&1 || true
docker network rm zlbench >/dev/null 2>&1 || true
echo "stopped"
