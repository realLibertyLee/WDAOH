"""Retrieve a checksum-bound AirData ZIP or verified E1a Parquet from the original HTTPS service."""

import csv
import datetime
import hashlib
import http.client
import os
import re
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "config/sources.csv"


def sources():
    """Return the eligible provider records, with byte counts, pollutant units and source receipts."""
    with CATALOG.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class _TLSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https" or target.username or target.password:
            raise ValueError("A redirect requires anonymous HTTPS")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def finish_short_read(opener, url, temp, size, expected, timeout, digest):
    """Resume a short response only at a provider-confirmed byte offset and vintage length."""
    requests = 0
    stalled = 0
    # Short ranges also work on routes that truncate the provider's larger responses.
    chunk_bytes = 64 << 10
    while size < expected:
        end = min(size + chunk_bytes, expected) - 1
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={size}-{end}",
                "Accept-Encoding": "identity",
                "User-Agent": "WDAOH/1.0",
            },
        )
        requests += 1
        if requests > 4 * (expected // chunk_bytes + 2):
            raise ValueError("Incomplete response after bounded range requests")
        try:
            with opener.open(request, timeout=timeout) as part:
                extent = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", part.headers.get("Content-Range", "")
                )
                if (
                    part.status != 206
                    or not extent
                    or tuple(map(int, extent.groups())) != (size, end, expected)
                ):
                    raise ValueError(
                        "Provider did not confirm the requested file range and length"
                    )
                try:
                    data = part.read(end - size + 1)
                except http.client.IncompleteRead as exc:
                    data = exc.partial
                if not data:
                    raise OSError("Truncated range response")
        except urllib.error.HTTPError:
            raise
        except (http.client.HTTPException, OSError) as exc:
            stalled += 1
            if stalled >= 5:
                raise ValueError("Repeated incomplete range response") from exc
            continue
        temp.write(data)
        digest.update(data)
        size += len(data)
        stalled = 0
    return size, requests


def acquire(item, folder, proxy=None, timeout=35):
    """Fetch one catalogued native file. Validate TLS, byte count, file schema and SHA before publishing locally.

    Input size is bytes and timeout is seconds per socket read. A cached match is labelled separately.
    Unavailable or revised vintages produce a structured receipt and no accepted replacement file.
    """
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    name = item["file"]
    url = urllib.parse.urlsplit(item["url"])
    if Path(name).name != name or url.scheme != "https" or url.username or url.password:
        raise ValueError("Unsafe file or source URL")
    target = folder / name
    expected = item["sha256"]
    expected_size = int(item["bytes"])
    record = dict(
        id=item["id"], url=item["url"], expected=expected, utc=start, timeout_s=timeout
    )
    if target.exists():
        h = hashlib.sha256()
        if target.is_symlink() or not target.is_file():
            raise ValueError("Unexpected local object: " + name)
        with target.open("rb") as stream:
            while buf := stream.read(1 << 20):
                h.update(buf)
        return dict(
            record,
            state="VERIFIED_EXISTING"
            if h.hexdigest() == expected and target.stat().st_size == expected_size
            else "LOCAL_FILE_MISMATCH",
            sha256=h.hexdigest(),
            bytes=target.stat().st_size,
        )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"https": proxy} if proxy else {})
        if proxy is not None
        else urllib.request.ProxyHandler(),
        _TLSRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    request = urllib.request.Request(
        item["url"], headers={"User-Agent": "WDAOH/1.0", "Accept-Encoding": "identity"}
    )
    try:
        with (
            opener.open(request, timeout=timeout) as response,
            tempfile.NamedTemporaryFile(dir=folder) as temp,
        ):
            h = hashlib.sha256()
            size = 0
            record.update(
                http=response.status,
                tls="verified",
                media=response.headers.get_content_type(),
                final_host=urllib.parse.urlsplit(response.geturl()).hostname,
            )
            while True:
                try:
                    chunk = response.read(64 << 10)
                except http.client.IncompleteRead as exc:
                    chunk = exc.partial
                except (http.client.HTTPException, OSError):
                    if not size:
                        raise
                    break
                if not chunk:
                    break
                h.update(chunk)
                size += len(chunk)
                if size > max(expected_size * 2, expected_size + (16 << 20)):
                    return dict(
                        record, state="UNEXPECTED_OVERSIZE", observed_at_least=size
                    )
                temp.write(chunk)
            if 0 < size < expected_size:
                record["initial_bytes"] = size
                size, record["range_requests"] = finish_short_read(
                    opener, item["url"], temp, size, expected_size, timeout, h
                )
            temp.flush()
            record.update(bytes=size, sha256=h.hexdigest())
            temp.seek(0)
            if item["domain"] == "EPA":
                if not zipfile.is_zipfile(temp.name):
                    return dict(record, state="INVALID_PROVIDER_FORMAT")
                with zipfile.ZipFile(temp.name) as z:
                    if item["member"] not in z.namelist():
                        return dict(record, state="UNEXPECTED_ZIP_MEMBER")
            else:
                magic = temp.read(4)
                if size < 8:
                    return dict(record, state="INVALID_PROVIDER_FORMAT")
                temp.seek(-4, 2)
                if magic != b"PAR1" or temp.read(4) != b"PAR1":
                    return dict(record, state="INVALID_PROVIDER_FORMAT")
            if record["sha256"] != expected or size != expected_size:
                return dict(record, state="SOURCE_VINTAGE_DRIFT")
            try:
                os.link(temp.name, target)
            except FileExistsError:
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or hashlib.sha256(target.read_bytes()).hexdigest() != expected
                ):
                    raise ValueError("Conflicting concurrent source file")
            return dict(record, state="VERIFIED_DOWNLOAD")
    except urllib.error.HTTPError as error:
        return dict(
            record,
            state="HTTP_ERROR",
            http=error.code,
            tls="verified",
            error=str(error),
        )
    except (
        OSError,
        http.client.HTTPException,
        ValueError,
        EOFError,
        zipfile.BadZipFile,
    ) as error:
        return dict(record, state="TRANSPORT_OR_FORMAT_ERROR", error=str(error))
