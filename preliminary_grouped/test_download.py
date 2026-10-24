import csv
import gzip
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import analyze


class FilterTests(unittest.TestCase):
    def connection(self, rows):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE raw(user_id TEXT, item_id TEXT, rating REAL, timestamp TEXT)")
        conn.executemany("INSERT INTO raw VALUES (?, ?, ?, ?)", rows)
        return conn

    def test_simultaneous_single_pass_keeps_cascading_low_degrees(self):
        pairs = [(f"u{u}", f"i{i}") for u in range(5) for i in range(5)]
        pairs += [("fragile_user", f"i{i}") for i in range(4)] + [("fragile_user", "leaf_item")]
        pairs += [(f"u{u}", "fragile_item") for u in range(4)] + [("leaf_user", "fragile_item")]
        conn = self.connection([(u, i, 4, None) for u, i in pairs])
        analyze.prepare(conn, 5)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM filtered").fetchone()[0], 33)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM filtered WHERE user_id='fragile_user'").fetchone()[0], 4)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM filtered WHERE item_id='fragile_item'").fetchone()[0], 4)

    def test_exactly_five_repeated_rows_and_pair_density(self):
        conn = self.connection([("u", "i", 5, None)] * 5)
        analyze.prepare(conn, 5)
        with tempfile.TemporaryDirectory() as directory:
            stats, _ = analyze.summarize(conn, "filtered", 5, Path(directory))
        self.assertEqual(stats["interactions"], 5)
        self.assertEqual(stats["unique_user_item_pairs"], 1)
        self.assertEqual(stats["repeated_pair_rows"], 4)
        self.assertEqual(stats["sparsity"], 0)

    def test_deduplicate_before_filter_and_earliest_known(self):
        conn = self.connection([("u", "i", 1, None), ("u", "i", 5, "2020-01-02"),
                                ("u", "i", 3, "2020-01-01")] + [("u", "i", 4, None)] * 2)
        analyze.prepare(conn, 5, True)
        self.assertEqual(conn.execute("SELECT rating FROM interactions").fetchone()[0], 3)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM filtered").fetchone()[0], 0)
        with tempfile.TemporaryDirectory() as directory:
            stats, _ = analyze.summarize(conn, "filtered", 5, Path(directory))
        self.assertIsNone(stats["sparsity"])
        self.assertIsNone(stats["degrees"]["user"]["median"])

    def test_degree_statistics(self):
        stats = analyze.degree_summary([(1, 2), (3, 1), (5, 1)], 5)
        self.assertEqual(stats["median"], 2)
        self.assertEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["gini"], .35)
        self.assertEqual(stats["top_1_percent_interaction_share"], .5)
        self.assertEqual(stats["below_threshold"], 3)


class AdapterTests(unittest.TestCase):
    def test_out_of_range_star_ratings_become_null_without_losing_interactions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reviews.jsonl"
            records = [dict(user_id="u", parent_asin="i", rating=rating,
                            timestamp=1609459200000) for rating in (0, 6, 5, 4, None)]
            source.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            result = analyze.analyze_dataset(dict(name="regression", format="amazon23",
                                                 paths=[source.name]), root, root / "out", export=True)
            self.assertEqual(result["after"]["interactions"], 5)
            self.assertEqual(result["after"]["missing_ratings"], 3)
            self.assertEqual(result["after"]["mean_rating"], 4.5)
            self.assertEqual(result["data_quality"]["out_of_range_star_ratings"], 2)
            self.assertEqual(result["data_quality"]["out_of_range_star_rating_values"], {"0.0": 1, "6.0": 1})
            with gzip.open(root / "out/regression/filtered_interactions.csv.gz", "rt") as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(len(exported), 5)
            self.assertEqual(exported[0]["rating"], "")
            analyze.report([result], root / "out")
            self.assertIn("interactions retained): 2", (root / "out/report.md").read_text(encoding="utf-8"))

    def test_amazon_parent_id_and_milliseconds(self):
        row = analyze.normalize(dict(user_id="001", asin="variant", parent_asin="parent",
                                     rating=4, timestamp=1609459200123), {"format": "amazon23"})
        self.assertEqual(row[:3], ("001", "parent", 4))
        self.assertEqual(row[3], "2021-01-01T00:00:00.123000+00:00")

    def test_yelp_and_movie_dates(self):
        yelp = analyze.normalize(dict(user_id="a", business_id="b", stars=5,
                                     date="2021-01-01 00:00:00"), {"format": "yelp"})
        movie = analyze.normalize(dict(user_id="a", item_id="b", rating=5,
                                      timestamp=1609459200), {"format": "movielens1m"})
        self.assertEqual(yelp, movie)
        self.assertEqual(analyze.timestamp("2021-01-01T08:00:00+08:00"), movie[3])

    def test_nested_steam_literals_and_gzip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steam.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(repr(dict(user_id="u", reviews=[
                    dict(item_id="g1", recommend=False, posted="Posted January 1, 2015."),
                    dict(item_id="g2", recommend=True, posted="Posted January 2.")])) + "\n")
            spec = {"format": "steam_user_reviews"}
            rows = [analyze.normalize(record, spec) for _, record in analyze.read_records(path, spec)]
        self.assertEqual(rows[0][2], 0)
        self.assertTrue(rows[0][3].startswith("2015-01-01"))
        self.assertEqual(rows[1], ("u", "g2", 1, None))

    def test_flat_steam_and_ownership(self):
        self.assertEqual(analyze.normalize(dict(username="u", product_id="g", date="2018-01-01"),
                                          {"format": "steam_reviews"})[:3], ("u", "g", None))
        self.assertEqual(analyze.normalize(dict(user_id="u", item_id="g", playtime_forever=0),
                                          {"format": "steam_items"}), ("u", "g", None, None))

    def test_invalid_values(self):
        for value in (None, "", True, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                analyze.identifier(value)
        with self.assertRaises(ValueError):
            analyze.number("nan")
        with self.assertRaises(ValueError):
            analyze.timestamp("not-a-date")

    def test_bad_line_has_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text("\nnot json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"bad.jsonl:2"):
                list(analyze.read_records(path, {"format": "amazon23"}))

    def test_config_names_cannot_escape_output(self):
        with self.assertRaises(ValueError):
            analyze.validate_config([dict(name="../escape", format="csv", paths=["x"])])


class IntegrationTests(unittest.TestCase):
    def test_all_configured_adapters_cli_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = []
            rows = {
                "amazon23": dict(user_id="u", parent_asin="i", rating=4, timestamp=1609459200000),
                "yelp": dict(user_id="u", business_id="i", stars=4, date="2021-01-01"),
                "steam_reviews": dict(username="u", product_id="i", date="2021-01-01"),
                "steam_user_reviews": dict(user_id="u", reviews=[dict(item_id="i", recommend=True, posted="Posted January 1, 2021.")]),
                "steam_items": dict(user_id="u", items=[dict(item_id="i", playtime_forever=0)]),
            }
            for kind, row in rows.items():
                path = root / f"{kind}.jsonl"
                path.write_text((json.dumps(row) + "\n") * 5, encoding="utf-8")
                specs.append(dict(name=kind, format=kind, paths=[path.name]))
            (root / "ratings.dat").write_text("1::2::4::1609459200\n" * 5, encoding="utf-8")
            specs.append(dict(name="movie", format="movielens1m", paths=["ratings.dat"]))
            (root / "data.csv").write_text("user_id,item_id,rating,timestamp\n" + "001,002,,\n" * 5, encoding="utf-8")
            specs.append(dict(name="csv", format="csv", paths=["data.csv"]))
            config = root / "config.json"
            config.write_text(json.dumps(specs), encoding="utf-8")
            output = root / "results"
            command = [sys.executable, str(Path(analyze.__file__)), "--config", str(config),
                       "--output", str(output), "--export"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for spec in specs:
                summary = json.loads((output / spec["name"] / "summary.json").read_text())
                self.assertEqual(summary["after"]["interactions"], 5)
                with gzip.open(output / spec["name"] / "filtered_interactions.csv.gz", "rt") as handle:
                    exported = list(csv.DictReader(handle))
                self.assertEqual(len(exported), 5)
            self.assertEqual(exported[0]["user_id"], "001")
            with (output / "comparison.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 14)
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertEqual(report.count("| after |"), 7)


if __name__ == "__main__":
    unittest.main()
