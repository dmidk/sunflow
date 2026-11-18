#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone
from io import BytesIO

import numpy as np
import pytest
import requests
import xarray as xr

from sunflow import downloaders
from sunflow.downloaders import (
    download_netcdf_dwd,
    download_netcdf_knmi,
    download_past_data,
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


class TestTransportFailures:
    """Network-level failures are not caught by the download helpers; they
    propagate to the callers, which decide whether to retry or skip."""

    @pytest.fixture(params=["knmi", "dwd"])
    def download(self, request, monkeypatch):
        """Call either downloader with `requests.get` raising `error`."""

        def _download(error):
            def fail(*args, **kwargs):
                raise error

            monkeypatch.setattr(requests, "get", fail)
            if request.param == "knmi":
                return download_netcdf_knmi("https://api.example.org/wcs", "key")
            return download_netcdf_dwd("https://opendata.dwd.de/file.nc")

        return _download

    def test_connection_errors_propagate(self, download):
        with pytest.raises(requests.exceptions.ConnectionError):
            download(requests.exceptions.ConnectionError("name resolution failed"))

    def test_timeouts_propagate(self, download):
        with pytest.raises(requests.exceptions.Timeout):
            download(requests.exceptions.Timeout("read timed out"))

    def test_chunked_encoding_errors_propagate(self, download):
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            download(requests.exceptions.ChunkedEncodingError("truncated response"))


class TestResponseContentIsNotValidated:
    """The helpers return whatever a 2xx response contained; parsing happens
    later, when xarray opens the buffer."""

    def test_an_empty_body_yields_an_empty_buffer(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda url: FakeResponse(b""))

        assert download_netcdf_dwd("https://opendata.dwd.de/file.nc").read() == b""

    def test_non_netcdf_content_is_returned_verbatim(self, monkeypatch):
        body = b"<html>503 Service Unavailable</html>"
        monkeypatch.setattr(requests, "get", lambda url: FakeResponse(body))

        assert download_netcdf_dwd("https://opendata.dwd.de/file.nc").read() == body

    def test_an_empty_knmi_body_yields_an_empty_buffer(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda url, headers: FakeResponse(b""))

        assert download_netcdf_knmi("https://api.example.org/wcs", "key").read() == b""


class TestDownloadPastDataSkipsFailures:
    """`download_past_data` logs and skips timesteps it cannot retrieve."""

    @pytest.fixture
    def time_steps(self):
        base = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        return [base + timedelta(minutes=15 * i) for i in range(3)]

    @pytest.fixture
    def dwd_config(self):
        return {"base_url": "https://opendata.dwd.de/", "variables": ["SIS", "SISc"]}

    def fake_file(self, time_step, value):
        """A DWD-shaped NetCDF buffer carrying its own time dimension."""
        dataset = xr.Dataset(
            {
                "SIS": (["lat", "lon"], np.full((8, 8), value)),
                "SISc": (["lat", "lon"], np.full((8, 8), 800.0)),
            },
            coords={"lat": np.arange(48.0, 56.0), "lon": np.arange(0.0, 8.0)},
        ).expand_dims(time=[time_step.replace(tzinfo=None)])
        buffer = BytesIO()
        dataset.to_netcdf(buffer, engine="h5netcdf")
        buffer.seek(0)
        return buffer

    def test_all_available_timesteps_are_returned(
        self, monkeypatch, time_steps, dwd_config
    ):
        available = {generate_dwd_filename(ts): ts for ts in time_steps}

        def fake_download(url):
            return self.fake_file(available[url.rsplit("/", 1)[-1]], 100.0)

        monkeypatch.setattr(downloaders, "download_netcdf_dwd", fake_download)

        ds = download_past_data(time_steps, dwd_config, "0,48,8,56", "DWD")

        assert ds.sizes["time"] == 3

    def test_a_failing_timestep_is_skipped_rather_than_aborting(
        self, monkeypatch, time_steps, dwd_config
    ):
        missing = generate_dwd_filename(time_steps[1])
        available = {generate_dwd_filename(ts): ts for ts in time_steps}

        def fake_download(url):
            filename = url.rsplit("/", 1)[-1]
            if filename == missing:
                raise requests.exceptions.HTTPError("404 Not Found")
            return self.fake_file(available[filename], 100.0)

        monkeypatch.setattr(downloaders, "download_netcdf_dwd", fake_download)

        ds = download_past_data(time_steps, dwd_config, "0,48,8,56", "DWD")

        assert ds.sizes["time"] == 2
        assert np.datetime64(time_steps[1].replace(tzinfo=None)) not in ds.time.values

    def test_an_empty_dataset_is_returned_when_everything_fails(
        self, monkeypatch, time_steps, dwd_config
    ):
        def fake_download(url):
            raise requests.exceptions.ConnectionError("network unreachable")

        monkeypatch.setattr(downloaders, "download_netcdf_dwd", fake_download)

        ds = download_past_data(time_steps, dwd_config, "0,48,8,56", "DWD")

        assert len(ds.data_vars) == 0
