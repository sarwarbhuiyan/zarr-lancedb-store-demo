#!/usr/bin/env bash
# LanceDB Enterprise (the `lancedb` CLI binary) over the benchmark's MinIO bucket:
#   qn  `lancedb server` with root s3://bench, external-Agent mode with no Agent started,
#       so nothing compacts, indexes or rewrites the benchmark tables
#   pe  `lancedb pe`, one plan executor with an explicit memory and disk page cache
# Both read MinIO through the toxiproxy latency proxy, so a cache miss pays the
# injected round trip and a cache hit does not.
#
# Usage: scripts/enterprise.sh start|stop|status|restart-pe [--cold]|logs [qn|pe]|wipe
#   restart-pe --cold   restart the PE with its memory and disk cache emptied
# Config: environment variables below. Needs ~/dev/lancedb-cli-helpers for the binary
# and its process helpers, and scripts/s3_up.sh running.
set -euo pipefail

LCLI_HELPERS="${LCLI_HELPERS:-$HOME/dev/lancedb-cli-helpers}"
DEMO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SETTINGS="${RUN_DIR:-$DEMO_ROOT/runs/enterprise}/settings.env"
# restart-pe reuses the settings the processes were started with, so a PE restarted from
# another shell keeps the same endpoint and cache sizes.
if [[ "${1:-}" != "start" && -f "$SETTINGS" ]]; then
  # shellcheck source=/dev/null
  source "$SETTINGS"
fi
: "${LANCEDB_CLI_TAG:=release-2026-10-06.ffb7c5b7fb.verified}"
: "${RUN_DIR:=$DEMO_ROOT/runs/enterprise}"
: "${QN_REST_PORT:=47410}"
: "${QN_SQL_PORT:=47411}"
: "${QN_MGMT_PORT:=47412}"
: "${PE_PORT:=47420}"
: "${API_KEY:=localkey}"
: "${ROOT_URI:=s3://bench}"
: "${S3_ENDPOINT:=http://127.0.0.1:19001}"          # toxiproxy in front of MinIO
: "${PE_MEMORY_CACHE_BYTES:=2147483648}"             # 2 GiB
: "${PE_DISK_CACHE_BYTES:=10737418240}"              # 10 GiB under $RUN_DIR/pe-cache
: "${PE_DISK_CACHE:=on}"                             # on | off
: "${EXTRA_QN_ARGS:=}"                               # e.g. --weak-read-consistency-interval-seconds 60
: "${EXTRA_PE_ARGS:=}"

# shellcheck source=/dev/null
source "$LCLI_HELPERS/lib/common.sh"

export AWS_ENDPOINT="$S3_ENDPOINT" AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 AWS_ALLOW_HTTP=true
export AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false AWS_EC2_METADATA_DISABLED=true

REST_URL="http://127.0.0.1:$QN_REST_PORT"
MGMT_URL="http://127.0.0.1:$QN_MGMT_PORT"

start_qn() {
  proc_running qn && return 0
  require_free_ports "$QN_REST_PORT" "$QN_SQL_PORT" "$QN_MGMT_PORT"
  # shellcheck disable=SC2086
  proc_start qn "$BIN" server \
    --host 127.0.0.1 --management-host 127.0.0.1 \
    --port "$QN_REST_PORT" --sql-port "$QN_SQL_PORT" --management-port "$QN_MGMT_PORT" \
    --metrics-port 0 --data-dir "$ROOT_URI" --api-key "$API_KEY" \
    --local-pes 0 --initial-agent-replicas 1 \
    --distributed-search=true --use-remote-scan=false \
    $EXTRA_QN_ARGS
  wait_port qn "$QN_MGMT_PORT"
}

start_pe() {
  proc_running pe && return 0
  require_free_ports "$PE_PORT"
  local cache_args=()
  if [[ "$PE_DISK_CACHE" == "on" ]]; then
    mkdir -p "$RUN_DIR/pe-cache"
    cache_args=(--disk-cache-dir "$RUN_DIR/pe-cache" --disk-cache-capacity "$PE_DISK_CACHE_BYTES")
  fi
  # shellcheck disable=SC2086
  proc_start pe "$BIN" pe \
    --management-url "$MGMT_URL" --port "$PE_PORT" --advertise-host 127.0.0.1 \
    --memory-cache-bytes "$PE_MEMORY_CACHE_BYTES" ${cache_args[@]+"${cache_args[@]}"} \
    $EXTRA_PE_ARGS
  wait_port pe "$PE_PORT"
  # The QN routes to a PE only after its health check passes; give registration a moment.
  sleep 3
}

BIN="$(resolve_cli "$LANCEDB_CLI_TAG")"

case "${1:-}" in
  start)
    mkdir -p "$RUN_DIR"
    for v in S3_ENDPOINT PE_MEMORY_CACHE_BYTES PE_DISK_CACHE_BYTES PE_DISK_CACHE EXTRA_QN_ARGS EXTRA_PE_ARGS; do
      printf '%s=%q\n' "$v" "${!v}"
    done >"$SETTINGS"
    start_qn
    start_pe
    wait_port qn "$QN_REST_PORT"
    echo "LanceDB $("$BIN" --version) over $ROOT_URI via $S3_ENDPOINT"
    echo "  REST $REST_URL   database: zarr   api key: $API_KEY"
    ;;
  stop)
    proc_stop pe
    proc_stop qn
    ;;
  status)
    proc_status qn "$QN_REST_PORT"
    proc_status pe "$PE_PORT"
    ;;
  restart-pe)
    proc_stop pe
    if [[ "${2:-}" == "--cold" ]]; then
      rm -rf "$RUN_DIR/pe-cache"
    fi
    start_pe
    ;;
  logs) tail -n "${LINES:-50}" -f "$(logfile "${2:-qn}")" ;;
  wipe)
    for p in qn pe; do
      proc_running "$p" && { echo "stop the processes before wiping" >&2; exit 1; }
    done
    rm -rf "$RUN_DIR" && echo "removed $RUN_DIR"
    ;;
  *) echo "usage: $0 start|stop|status|restart-pe [--cold]|logs [qn|pe]|wipe" >&2; exit 2 ;;
esac
