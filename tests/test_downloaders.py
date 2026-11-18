#!/usr/bin/env python3
from datetime import datetime, timezone
from io import BytesIO

import pytest
import requests

from sunflow.downloaders import (
    download_netcdf_dwd,
    download_netcdf_knmi,
    generate_dwd_filename,
)


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, content: bytes = b"", status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class TestGenerateDwdFilename:
    def test_follows_the_dwd_naming_convention(self):
        timestamp = datetime(2025, 1, 15, 12, 30, tzinfo=timezone.utc)

        assert generate_dwd_filename(timestamp) == "SISin202501151230FDv3.nc"

    def test_date_parts_are_zero_padded(self):
        timestamp = datetime(2026, 3, 9, 4, 5, tzinfo=timezone.utc)

        assert generate_dwd_filename(timestamp) == "SISin202603090405FDv3.nc"

    def test_seconds_are_not_part_of_the_filename(self):
        with_seconds = datetime(2026, 3, 9, 12, 30, 45, tzinfo=timezone.utc)
        without_seconds = datetime(2026, 3, 9, 12, 30, tzinfo=timezone.utc)

        assert generate_dwd_filename(with_seconds) == generate_dwd_filename(
            without_seconds
        )


class TestDownloadNetcdfKnmi:
    def test_content_is_returned_as_a_buffer(self, monkeypatch):
        monkeypatch.setattr(
            requests, "get", lambda url, headers: FakeResponse(b"netcdf-bytes")
        )

        buffer = download_netcdf_knmi("https://api.example.org/wcs", "secret-key")

        assert isinstance(buffer, BytesIO)
        assert buffer.read() == b"netcdf-bytes"

    def test_the_api_key_is_sent_as_an_authorization_header(self, monkeypatch):
        captured = {}

        def fake_get(url, headers):
            captured["url"] = url
            captured["headers"] = headers
            return FakeResponse(b"")

        monkeypatch.setattr(requests, "get", fake_get)

        download_netcdf_knmi("https://api.example.org/wcs", "secret-key")

        assert captured["headers"] == {"Authorization": "secret-key"}
        assert captured["url"] == "https://api.example.org/wcs"

    @pytest.mark.parametrize("status_code", [401, 404, 500])
    def test_error_responses_raise(self, monkeypatch, status_code):
        monkeypatch.setattr(
            requests, "get", lambda url, headers: FakeResponse(b"", status_code)
        )

        with pytest.raises(requests.exceptions.HTTPError):
            download_netcdf_knmi("https://api.example.org/wcs", "secret-key")


class TestDownloadNetcdfDwd:
    def test_content_is_returned_as_a_buffer(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda url: FakeResponse(b"netcdf-bytes"))

        buffer = download_netcdf_dwd("https://opendata.dwd.de/file.nc")

        assert buffer.read() == b"netcdf-bytes"

    def test_no_authentication_header_is_sent(self, monkeypatch):
        captured = {}

        def fake_get(url):
            captured["url"] = url
            return FakeResponse(b"")

        monkeypatch.setattr(requests, "get", fake_get)

        download_netcdf_dwd("https://opendata.dwd.de/file.nc")

        assert captured["url"] == "https://opendata.dwd.de/file.nc"

    @pytest.mark.parametrize("status_code", [404, 500])
    def test_error_responses_raise(self, monkeypatch, status_code):
        monkeypatch.setattr(requests, "get", lambda url: FakeResponse(b"", status_code))

        with pytest.raises(requests.exceptions.HTTPError):
            download_netcdf_dwd("https://opendata.dwd.de/file.nc")
