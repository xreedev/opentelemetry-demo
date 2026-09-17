"""Uploads catalog/*.yaml fault entries into Elasticsearch as a
semantic_text-indexed index, so they can be vector/semantic-searched
(e.g. "queue is backed up and consumer lag is spiking" -> kafkaQueueProblems).

Targets either:
  - Elastic Cloud / Serverless (the web console's deployment) via
    ES_URL + ES_API_KEY environment variables, same convention as
    backend/app/clients/elastic.py's `ApiKey` auth; or
  - the local self-hosted stack at localhost:9200 (probe-detector/
    es_client.py's convention), used automatically when ES_URL/ES_API_KEY
    are not set, with the password read from
    opentelemetry-demo/elastic-start-local/.env.

Usage:
    pip install pyyaml   # only non-stdlib dependency

    # Elastic Cloud / Serverless:
    export ES_URL="https://<your-deployment>.es.<region>.cloud.es.io"
    export ES_API_KEY="<api key from Kibana Stack Management > API keys>"
    python catalog/upload_to_es.py

    # local self-hosted stack (default, no env vars needed):
    python catalog/upload_to_es.py
"""
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = REPO_ROOT / "catalog"
START_LOCAL_ENV = REPO_ROOT / "opentelemetry-demo" / "elastic-start-local" / ".env"
INDEX_NAME = "fault-catalog"
# On Elastic Cloud/Serverless the preconfigured ELSER endpoint id can differ
# (e.g. ".elser-2-elastic" on Serverless) -- override with ES_INFERENCE_ID
# if `ensure_inference_endpoint` can't find/create the default below.
INFERENCE_ID = os.environ.get("ES_INFERENCE_ID", ".elser-2-elasticsearch")


def _es_password() -> str:
    for line in START_LOCAL_ENV.read_text().splitlines():
        if line.startswith("ES_LOCAL_PASSWORD="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ES_LOCAL_PASSWORD not found")


ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
_api_key = os.environ.get("ES_API_KEY")
_AUTH_HEADER = f"ApiKey {_api_key}" if _api_key else (
    "Basic " + base64.b64encode(f"elastic:{_es_password()}".encode()).decode()
)


def _request(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{ES_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": _AUTH_HEADER},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode()}") from e


def ensure_inference_endpoint() -> None:
    """Confirm the ELSER inference endpoint semantic_text will use exists."""
    try:
        _request("GET", f"/_inference/{INFERENCE_ID}")
    except RuntimeError as e:
        if "404" not in str(e):
            raise
        print(f"Creating inference endpoint {INFERENCE_ID} ...")
        _request(
            "PUT",
            f"/_inference/sparse_embedding/{INFERENCE_ID}",
            {"service": "elasticsearch", "service_settings": {"num_allocations": 1, "num_threads": 1}},
        )


def ensure_index() -> None:
    existing = _request("GET", "/_cat/indices?format=json")
    if any(idx.get("index") == INDEX_NAME for idx in existing):
        print(f"Index {INDEX_NAME} already exists, reusing it.")
        return
    mapping = {
        "mappings": {
            "properties": {
                "flag": {"type": "keyword"},
                "enabled_value": {"type": "keyword"},
                "fault_class": {"type": "keyword"},
                "service": {"type": "keyword"},
                "loudest_service": {"type": "keyword"},
                "change_point": {"type": "keyword"},
                "metric": {"type": "keyword"},
                "hops_to_cause": {"type": "integer"},
                "dependency": {"type": "keyword"},
                "drift": {"type": "boolean"},
                "permanent_fix": {"type": "text"},
                "steps": {"type": "text"},
                "content": {
                    "type": "semantic_text",
                    "inference_id": INFERENCE_ID,
                },
            }
        }
    }
    _request("PUT", f"/{INDEX_NAME}", mapping)
    print(f"Created index {INDEX_NAME}.")


def load_catalog_docs() -> list[dict]:
    docs = []
    for path in sorted(CATALOG_DIR.glob("*.yaml")):
        entry = yaml.safe_load(path.read_text())
        truth = entry["truth"]
        sig = entry["expected_signature"]
        runbook = entry["reference_runbook"]
        content = "\n".join(
            [
                f"Flag: {entry['flag']}",
                f"Cause: {runbook['cause']}",
                f"Fault class: {truth['fault_class']}",
                f"Notes: {entry.get('notes', '')}",
            ]
        )
        docs.append(
            {
                "flag": entry["flag"],
                "enabled_value": entry["enabled_value"],
                "fault_class": truth["fault_class"],
                "service": truth["service"],
                "loudest_service": sig["loudest_service"],
                "change_point": sig["change_point"],
                "metric": sig["metric"],
                "hops_to_cause": sig["hops_to_cause"],
                "dependency": sig.get("dependency"),
                "drift": entry["drift"],
                "permanent_fix": runbook["permanent_fix"],
                "steps": " ".join(runbook["steps"]),
                "content": content,
            }
        )
    return docs


def bulk_upload(docs: list[dict]) -> None:
    lines = []
    for doc in docs:
        lines.append(json.dumps({"index": {"_index": INDEX_NAME, "_id": doc["flag"]}}))
        lines.append(json.dumps(doc))
    body = "\n".join(lines) + "\n"
    req = urllib.request.Request(
        f"{ES_URL}/_bulk",
        data=body.encode(),
        headers={"Content-Type": "application/x-ndjson", "Authorization": _AUTH_HEADER},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    errors = [item for item in result["items"] if item["index"].get("error")]
    if errors:
        raise RuntimeError(f"{len(errors)} documents failed: {errors[:3]}")
    print(f"Indexed {len(docs)} documents into {INDEX_NAME}.")


if __name__ == "__main__":
    ensure_inference_endpoint()
    ensure_index()
    bulk_upload(load_catalog_docs())
