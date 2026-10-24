import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import download_datasets as downloader


class Response:
    def __init__(self, data, start, end, total, status=206):
        self.data = data
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.data), chunk_size):
            yield self.data[offset:offset + chunk_size]


class DownloadTests(unittest.TestCase):
    def test_parallel_ranges_preserve_prefix_and_merge_exact_bytes(self):
        data = bytes(range(256)) * (129 * 1024)
        requests = []

        def get(url, headers, **kwargs):
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            requests.append((start, end))
            return Response(data[start:end + 1], start, end, len(data))

        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "dataset.part"
            partial.write_bytes(data[:123])
            with patch.object(downloader.requests, "get", side_effect=get):
                downloader.download_ranges("https://example.test/data", partial, len(data), 2, "fixture")
            self.assertEqual(partial.read_bytes(), data)
            self.assertEqual(len(requests), 2)
            self.assertEqual(min(start for start, _ in requests), 123)
            self.assertFalse(partial.with_name("dataset.part.chunks").exists())

    def test_reject_server_ignoring_range(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "dataset.part"
            response = Response(b"abc", 0, 2, 3, status=200)
            with patch.object(downloader.requests, "get", return_value=response):
                with self.assertRaisesRegex(ValueError, "inconsistent byte range"):
                    downloader.download_ranges("https://example.test/data", partial, 3, 2, "fixture")
            self.assertFalse(partial.exists())

    def test_yelp_archive_extracts_only_to_fixed_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = (json.dumps(dict(user_id="u", business_id="i", stars=5, date="2020-01-01")) + "\n").encode()
            archive = root / "dataset.tar"
            with tarfile.open(archive, "w") as handle:
                member = tarfile.TarInfo("../../yelp_academic_dataset_review.json")
                member.size = len(content)
                handle.addfile(member, io.BytesIO(content))
            downloader.import_yelp(archive, root / "data")
            self.assertEqual((root / "data/yelp/yelp_academic_dataset_review.json").read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
