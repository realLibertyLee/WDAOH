"""Download study measurements from the official provider archive."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wdaoh.data.providers import acquire, sources


def download(provider: str, pollutant: str, raw: Path) -> list[dict]:
    """Verify each selected native file and keep request receipts alongside the raw measurements."""
    catalog = [row for row in sources() if row["domain"] == provider]
    if provider == "EPA" and pollutant != "all":
        catalog = [row for row in catalog if row["pollutant"] == pollutant]
    if provider == "EEA" and pollutant != "all":
        raise ValueError(
            "Use all EEA pollutants to retain every colocated supplier and sampling-point decision."
        )
    if not catalog:
        raise ValueError("Empty provider selection")
    raw.mkdir(exist_ok=True, parents=True)

    def fetch(row):
        attempts = []
        fallback = [
            v for v in os.environ.get("WDAOH_FALLBACK_PROXIES", "").split(",") if v
        ]
        for proxy in [None, *fallback[:2]]:
            result = acquire(row, raw, proxy=proxy)
            attempts.append(result.copy())
            if result["state"] not in (
                "TRANSPORT_OR_FORMAT_ERROR",
                "INVALID_PROVIDER_FORMAT",
            ):
                break
        result["attempts"] = attempts
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(fetch, catalog))
    receipt = raw / f"{provider}_{pollutant}_download.json"
    receipt.write_text(json.dumps(receipts, indent=2) + "\n")
    failed = [
        r
        for r in receipts
        if r["state"] not in {"VERIFIED_DOWNLOAD", "VERIFIED_EXISTING"}
    ]
    if failed:
        raise ValueError(
            f"{len(failed)}/{len(catalog)} sources failed verification. See {receipt}."
        )
    print(
        f"{provider} {pollutant}: {len(receipts)} verified provider files", flush=True
    )
    return receipts
