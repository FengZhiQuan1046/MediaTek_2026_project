"""Explore history lengths and temporal patterns in existing filtered exports."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from analyze import degree_summary, write_csv


BUCKETS = [("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10-19", 10, 19),
           ("20-49", 20, 49), ("50-99", 50, 99), ("100+", 100, None)]


def ratio(a, b):
    return a / b if b else None


def continuous_stats(conn, table, field):
    n, low, mean, high = conn.execute(
        f"SELECT COUNT({field}), MIN({field}), AVG({field}), MAX({field}) FROM {table}").fetchone()
    result = dict(count=n, min=low, mean=mean, max=high)
    quantiles = {"p25": .25, "median": .5, "p75": .75, "p90": .9, "p95": .95, "p99": .99}
    if not n:
        return {**result, **dict.fromkeys(quantiles)}
    ranks = {key: (n - 1) * q for key, q in quantiles.items()}
    needed = {r for value in ranks.values() for r in (math.floor(value), math.ceil(value))}
    selected = {}
    cursor = conn.execute(f"SELECT {field} FROM {table} WHERE {field} IS NOT NULL ORDER BY {field}")
    for index, (value,) in enumerate(cursor):
        if index in needed:
            selected[index] = value
        if index >= max(needed):
            break
    cursor.close()
    for key, rank in ranks.items():
        floor = math.floor(rank)
        result[key] = selected[floor] + (selected[math.ceil(rank)] - selected[floor]) * (rank - floor)
    return result


def load_export(conn, path):
    conn.execute("CREATE TABLE events(user_id TEXT, item_id TEXT, ts REAL, day INTEGER)")
    batch = []
    n = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"user_id", "item_id", "timestamp_utc"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected filtered interaction export columns")
        for record in reader:
            if not record["user_id"] or not record["item_id"]:
                raise ValueError(f"{path}:{reader.line_num}: empty user or item ID")
            value = record["timestamp_utc"]
            ts = None
            if value:
                date = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if date.tzinfo is None:
                    raise ValueError(f"{path}:{reader.line_num}: timezone required")
                ts = date.timestamp()
            batch.append((record["user_id"], record["item_id"], ts,
                          math.floor(ts / 86400) if ts is not None else None))
            n += 1
            if len(batch) == 10000:
                conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", batch)
                batch.clear()
            if n % 1000000 == 0:
                print(f"  loaded {n:,} filtered interactions", flush=True)
    conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", batch)
    conn.commit()
    return n


def compressed_csv(conn, path, query):
    cursor = conn.execute(query)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([column[0] for column in cursor.description])
        writer.writerows(cursor)


def history_metrics(conn, source, out):
    k = source["filter"]["threshold"]
    print("  computing user histories and distinct items", flush=True)
    conn.execute("""CREATE TABLE users AS SELECT user_id, COUNT(*) AS n,
        COUNT(DISTINCT item_id) AS distinct_items, COUNT(ts) AS dated_events,
        COUNT(DISTINCT day) AS active_days, MIN(ts) AS first_ts, MAX(ts) AS last_ts,
        (MAX(ts)-MIN(ts))/86400.0 AS span_days FROM events GROUP BY user_id""")
    print("  computing item audiences", flush=True)
    conn.execute("""CREATE TABLE items AS SELECT item_id, COUNT(*) AS n,
        COUNT(DISTINCT user_id) AS distinct_users, COUNT(ts) AS dated_events,
        (MAX(ts)-MIN(ts))/86400.0 AS span_days FROM events GROUP BY item_id""")
    n, users, items, pairs = conn.execute("""SELECT (SELECT COUNT(*) FROM events),
        (SELECT COUNT(*) FROM users), (SELECT COUNT(*) FROM items),
        (SELECT COALESCE(SUM(distinct_items),0) FROM users)""").fetchone()
    for key, actual in {"interactions": n, "users": users, "items": items,
                        "unique_user_item_pairs": pairs}.items():
        if actual != source["after"][key]:
            raise ValueError(f"export/summary mismatch for {key}: {actual} vs {source['after'][key]}")
    distributions = {}
    for label, table, field in (("user_events", "users", "n"),
                                 ("user_distinct_items", "users", "distinct_items"),
                                 ("item_events", "items", "n"),
                                 ("item_distinct_users", "items", "distinct_users")):
        hist = list(conn.execute(f"SELECT {field}, COUNT(*) FROM {table} GROUP BY {field} ORDER BY {field}"))
        distributions[label] = degree_summary(hist, k)
        write_csv(out / f"{label}.csv", ["length", "entities"], hist)

    print("  measuring chronological gaps and timestamp ties", flush=True)
    # Null dates cannot define chronology. Source rowid only resolves ties deterministically;
    # item-to-item adjacency is reported exclusively across strictly positive gaps.
    conn.execute("""CREATE TABLE gaps AS SELECT user_id, seq, ts, ts-previous_ts AS seconds,
        item_id=previous_item AS same_item FROM (
          SELECT rowid AS seq, user_id, item_id, ts,
          LAG(ts) OVER w AS previous_ts, LAG(item_id) OVER w AS previous_item
          FROM events WHERE ts IS NOT NULL
          WINDOW w AS (PARTITION BY user_id ORDER BY ts, rowid)
        ) WHERE previous_ts IS NOT NULL""")
    conn.execute("""CREATE TABLE user_gaps AS SELECT user_id, COUNT(*) AS observed_gaps,
        SUM(seconds=0) AS tied_gaps, AVG(seconds)/86400.0 AS mean_gap_days,
        MAX(seconds)/86400.0 AS max_gap_days FROM gaps GROUP BY user_id""")
    print("  grouping reviews into clusters separated by over 30 days", flush=True)
    conn.execute("""CREATE TABLE review_clusters AS SELECT user_id, cluster_id, COUNT(*) AS n,
        MIN(ts) AS first_ts, MAX(ts) AS last_ts, (MAX(ts)-MIN(ts))/86400.0 AS span_days FROM (
          SELECT user_id, ts, SUM(new_cluster) OVER (
            PARTITION BY user_id ORDER BY ts,seq ROWS UNBOUNDED PRECEDING) AS cluster_id FROM (
              SELECT user_id, first_ts AS ts, 0 AS seq, 0 AS new_cluster
              FROM users WHERE dated_events>0
              UNION ALL SELECT user_id,ts,seq,seconds>2592000 AS new_cluster FROM gaps))
        GROUP BY user_id,cluster_id""")
    conn.execute("""CREATE TABLE user_clusters AS SELECT c.user_id, COUNT(*) AS clusters,
        MAX(c.n) AS largest_cluster_events, MAX(c.n)*1.0/u.dated_events AS largest_cluster_share
        FROM review_clusters c JOIN users u USING(user_id) GROUP BY c.user_id""")
    cluster_hist = list(conn.execute("SELECT n,COUNT(*) FROM review_clusters GROUP BY n ORDER BY n"))
    write_csv(out / "cluster_sizes.csv", ["events", "clusters"], cluster_hist)
    compressed_csv(conn, out / "review_clusters.csv.gz", """SELECT user_id,cluster_id,n AS events,
        first_ts AS first_timestamp_epoch_seconds,last_ts AS last_timestamp_epoch_seconds,
        span_days FROM review_clusters""")
    cluster_count_stats = continuous_stats(conn, "user_clusters", "clusters")
    cluster_share_stats = continuous_stats(conn, "user_clusters", "largest_cluster_share")
    cluster_span_stats = continuous_stats(conn, "review_clusters", "span_days")
    cluster_single, eligible_cluster_users = conn.execute("""SELECT COALESCE(SUM(c.clusters=1),0),COUNT(*)
        FROM user_clusters c JOIN users u USING(user_id) WHERE u.dated_events>=2""").fetchone()
    sensitivity = []
    for days in (1, 7, 30, 90):
        # Group once per threshold; avoid one correlated scan for every user.
        separated = conn.execute("SELECT COUNT(DISTINCT user_id) FROM gaps WHERE seconds>?", (days*86400,)).fetchone()[0]
        sensitivity.append(dict(inactivity_days=days,
                                multi_dated_user_one_cluster_share=ratio(eligible_cluster_users-separated,eligible_cluster_users)))
    edges, tied, day, week, month, positive, same_item = conn.execute("""SELECT COUNT(*),
        COALESCE(SUM(seconds=0),0), COALESCE(SUM(seconds>0 AND seconds<=86400),0),
        COALESCE(SUM(seconds>0 AND seconds<=604800),0),
        COALESCE(SUM(seconds>2592000),0), COALESCE(SUM(seconds>0),0),
        COALESCE(SUM(seconds>0 AND same_item),0) FROM gaps""").fetchone()
    missing, multi_dated, single_day, midnight = conn.execute("""SELECT
        (SELECT COUNT(*) FROM events WHERE ts IS NULL),
        (SELECT COUNT(*) FROM users WHERE dated_events>=2),
        (SELECT COUNT(*) FROM users WHERE dated_events>=2 AND active_days=1),
        (SELECT COUNT(*) FROM events WHERE ts IS NOT NULL AND ts=day*86400)""").fetchone()
    users_with_repeats = conn.execute("SELECT COUNT(*) FROM users WHERE n>distinct_items").fetchone()[0]
    # Repeated dated rows of the same pair and exact timestamp are particularly ambiguous.
    same_pair_ties = conn.execute("""SELECT COALESCE(SUM(n-1),0) FROM (
        SELECT COUNT(*) AS n FROM events WHERE ts IS NOT NULL GROUP BY user_id,item_id,ts)""").fetchone()[0]

    bucket_rows = []
    for label, lower, upper in BUCKETS:
        clause = "n>=?" if upper is None else "n BETWEEN ? AND ?"
        params = (lower,) if upper is None else (lower, upper)
        count, events, avg_distinct, avg_span, avg_days = conn.execute(f"""SELECT COUNT(*),
            COALESCE(SUM(n),0), AVG(distinct_items), AVG(span_days), AVG(active_days)
            FROM users WHERE {clause}""", params).fetchone()
        bucket_rows.append([label, count, ratio(count, users), events, ratio(events, n),
                            avg_distinct, avg_span, avg_days])
    write_csv(out / "length_buckets.csv", ["history_events", "users", "user_share", "events",
              "event_share", "mean_distinct_items", "mean_span_days", "mean_active_days"], bucket_rows)
    capacity = []
    for length in (5, 10, 20, 50, 100, 200):
        count, lost = conn.execute("SELECT COUNT(*), COALESCE(SUM(n-?),0) FROM users WHERE n>?",
                                  (length, length)).fetchone()
        capacity.append(dict(max_length=length, users_truncated=count, user_share=ratio(count, users),
                             events_removed=lost, event_share_removed=ratio(lost, n)))
    compressed_csv(conn, out / "user_histories.csv.gz", """SELECT u.user_id, u.n AS interactions,
        u.distinct_items, u.n-u.distinct_items AS repeated_pair_rows, u.dated_events, u.active_days,
        u.span_days, u.first_ts AS first_timestamp_epoch_seconds,
        u.last_ts AS last_timestamp_epoch_seconds, COALESCE(g.observed_gaps,0) AS observed_gaps,
        COALESCE(g.tied_gaps,0) AS tied_gaps, g.mean_gap_days, g.max_gap_days
        , c.clusters AS clusters_30_days, c.largest_cluster_events, c.largest_cluster_share
        FROM users u LEFT JOIN user_gaps g USING(user_id) LEFT JOIN user_clusters c USING(user_id)""")
    compressed_csv(conn, out / "item_histories.csv.gz", """SELECT item_id, n AS interactions,
        distinct_users, n-distinct_users AS repeated_pair_rows, dated_events, span_days FROM items""")
    gap_buckets = [("same timestamp", "seconds=0"), ("0-1 hour", "seconds>0 AND seconds<=3600"),
                   ("1 hour-1 day", "seconds>3600 AND seconds<=86400"),
                   ("1-7 days", "seconds>86400 AND seconds<=604800"),
                   ("7-30 days", "seconds>604800 AND seconds<=2592000"),
                   ("30-90 days", "seconds>2592000 AND seconds<=7776000"),
                   ("90-365 days", "seconds>7776000 AND seconds<=31536000"),
                   ("over 365 days", "seconds>31536000")]
    gap_rows = [(label, conn.execute(f"SELECT COUNT(*) FROM gaps WHERE {where}").fetchone()[0])
                for label, where in gap_buckets]
    write_csv(out / "gap_buckets.csv", ["gap", "adjacent_events"], gap_rows)
    positive_stats = continuous_stats(conn, "(SELECT seconds/86400.0 AS days FROM gaps WHERE seconds>0)", "days")
    span_stats = continuous_stats(conn, "users", "span_days")
    active_day_stats = continuous_stats(conn, "users", "active_days")
    save_timeline_samples(conn, out)
    result = dict(dataset=source["dataset"], source_filter=source["filter"],
                  source_summary_generated_at_utc=source["generated_at_utc"],
                  interactions=n, users=users, items=items, distributions=distributions,
                  repeated_pair_rows=n-pairs, repeated_event_share=ratio(n-pairs, n),
                  users_with_repeats=users_with_repeats, users_with_repeats_share=ratio(users_with_repeats, users),
                  same_pair_timestamp_excess_rows=same_pair_ties,
                  missing_timestamps=missing, observed_adjacent_gaps=edges,
                  tied_adjacent_gaps=tied, tied_gap_share=ratio(tied, edges),
                  positive_adjacent_gaps=positive, positive_gap_days=positive_stats,
                  positive_gaps_within_day=day, positive_gaps_within_week=week,
                  positive_gaps_over_30_days=month,
                  adjacent_same_item_across_positive_gaps=same_item,
                  users_with_at_least_two_dated_events=multi_dated,
                  multi_dated_users_with_one_active_day=single_day,
                  single_active_day_share=ratio(single_day, multi_dated),
                  midnight_timestamp_share=ratio(midnight, n-missing),
                  user_span_days=span_stats, user_active_days=active_day_stats,
                  review_clusters_30_days=dict(cluster_count=cluster_count_stats,
                      largest_cluster_share=cluster_share_stats,
                      cluster_span_days=cluster_span_stats,
                      cluster_event_counts=degree_summary(cluster_hist, k),
                      multi_dated_user_one_cluster_share=ratio(cluster_single,eligible_cluster_users),
                      threshold_sensitivity=sensitivity),
                  users_below_5=conn.execute("SELECT COUNT(*) FROM users WHERE n<5").fetchone()[0],
                  users_with_at_least_3_events=conn.execute("SELECT COUNT(*) FROM users WHERE n>=3").fetchone()[0],
                  users_with_at_least_3_distinct_items=conn.execute("SELECT COUNT(*) FROM users WHERE distinct_items>=3").fetchone()[0],
                  truncation=capacity)
    conn.commit()
    return result


def save_timeline_samples(conn, out):
    count = conn.execute("SELECT COUNT(*) FROM users WHERE dated_events>=2").fetchone()[0]
    samples = []
    if count:
        ranks = {round((count-1)*q):q for q in (.1,.25,.5,.75,.9,.95,.99,1)}
        cursor = conn.execute("""SELECT u.user_id,u.n,u.dated_events,u.span_days,c.clusters
            FROM users u JOIN user_clusters c USING(user_id) WHERE dated_events>=2
            ORDER BY u.n,u.user_id""")
        for rank, row in enumerate(cursor):
            if rank in ranks:
                samples.append((f"P{100*ranks[rank]:g}",*row))
        cursor.close()
    write_csv(out / "timeline_users.csv", ["length_percentile", "user_id", "events", "dated_events", "span_days", "clusters_30_days"], samples)
    with gzip.open(out / "timeline_events.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "timestamp_epoch_seconds"])
        if samples:
            ids = [s[1] for s in samples]
            marks = ",".join("?" for _ in ids)
            writer.writerows(conn.execute(f"SELECT user_id,ts FROM events WHERE ts IS NOT NULL AND user_id IN ({marks}) ORDER BY user_id,ts",ids))


def run_dataset(folder, output, work_dir=None):
    source = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    out = output / folder.name
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="histories-", dir=work_dir) as scratch:
        with closing(sqlite3.connect(str(Path(scratch) / "histories.sqlite"))) as conn:
            conn.execute("PRAGMA temp_store=FILE")
            conn.execute("PRAGMA cache_size=-65536")
            n = load_export(conn, folder / "filtered_interactions.csv.gz")
            if n != source["after"]["interactions"]:
                raise ValueError(f"{folder}: export row count differs from summary")
            result = history_metrics(conn, source, out)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "history_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def fmt(value, percent=False):
    if value is None:
        return "N/A"
    return f"{value:.1%}" if percent else f"{value:,.2f}"


def write_report(results, output):
    columns = ["dataset", "users", "mean_events", "median_events", "p95_events", "max_events",
               "median_distinct_items", "users_below_5_share", "repeated_event_share",
               "tied_gap_share", "median_positive_gap_days", "median_user_span_days", "item_top_1_percent_event_share"]
    rows = []
    lines = ["# User-item interaction histories", "",
             "Computed from the existing filtered exports, without filtering again. A history is a user's retained review/rating rows, ordered by observed timestamp. It is not a purchase, play, or browsing log.", "",
             "## Lengths and distinct items", "",
             "| Dataset | Users | Mean events | Median | P95 | Max | Median distinct items | Users <5 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        d = result["distributions"]
        event, distinct = d["user_events"], d["user_distinct_items"]
        rows.append([result["dataset"], result["users"], event["mean"], event["median"], event["p95"], event["max"],
                     distinct["median"], ratio(result["users_below_5"], result["users"]), result["repeated_event_share"],
                     result["tied_gap_share"], result["positive_gap_days"]["median"], result["user_span_days"]["median"],
                     d["item_events"]["top_1_percent_interaction_share"]])
        lines.append(f"| {result['dataset']} | {result['users']:,} | {fmt(event['mean'])} | {fmt(event['median'])} | {fmt(event['p95'])} | {event['max']} | {fmt(distinct['median'])} | {fmt(ratio(result['users_below_5'],result['users']),True)} |")
    lines.extend(["", "## Repetition and timing", "",
                  "Repeated-event share is (rows minus distinct user-item pairs) / rows. Tied-gap share is the fraction of adjacent dated events with identical timestamps. Positive-gap medians exclude ties and weight every gap equally, not every user.", "",
                  "| Dataset | Repeated events | Users with repeat pairs | Tied gaps | Median positive gap (days) | Median history span (days) | One-day users* | Top 1% items' event share |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for r in results:
        lines.append(f"| {r['dataset']} | {fmt(r['repeated_event_share'],True)} | {fmt(r['users_with_repeats_share'],True)} | {fmt(r['tied_gap_share'],True)} | {fmt(r['positive_gap_days']['median'])} | {fmt(r['user_span_days']['median'])} | {fmt(r['single_active_day_share'],True)} | {fmt(r['distributions']['item_events']['top_1_percent_interaction_share'],True)} |")
    lines.extend(["", "*Among users with at least two dated events. Active days are UTC calendar dates. History spans describe the observed window, not tenure; single-dated-event users have zero span. Missing timestamps are excluded from gaps and active-day counts.", "",
                  "## Length caps", "",
                  "Share of retained events removed by limiting each user to at most K rows. These are capacity calculations, without constructing train/test splits.", "",
                  "| Dataset | K=10 | K=20 | K=50 | K=100 |", "| --- | ---: | ---: | ---: | ---: |"])
    for r in results:
        caps = {c["max_length"]: c for c in r["truncation"]}
        lines.append("| " + r["dataset"] + " | " + " | ".join(fmt(caps[k]["event_share_removed"], True) for k in (10,20,50,100)) + " |")
    lines.extend(["", "## Chronological clustering", "",
                  "A cluster joins consecutive dated reviews from the same user while every gap is at most 30 days. A gap strictly greater than 30 days starts another cluster. A cluster may span longer than 30 days through a chain of reviews; it is not a fixed 30-day window or a browsing session. Cluster proportions use dated events only.", "",
                  "| Dataset | Median clusters/user* | Users in one cluster** | Median largest-cluster share* | Median cluster span (days) | P95 cluster span (days) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for r in results:
        c = r["review_clusters_30_days"]
        lines.append(f"| {r['dataset']} | {fmt(c['cluster_count']['median'])} | {fmt(c['multi_dated_user_one_cluster_share'],True)} | {fmt(c['largest_cluster_share']['median'],True)} | {fmt(c['cluster_span_days']['median'])} | {fmt(c['cluster_span_days']['p95'])} |")
    lines.extend(["", "*Among users with at least one dated event. **Among users with at least two dated events.", "",
                  "The threshold is descriptive. The JSON summaries also provide one-cluster shares with 1-, 7-, and 90-day inactivity thresholds. Singleton clusters have zero span, so median spans can be zero even when users have multi-year histories.", "",
                  "## Sensitivity to the inactivity threshold", "",
                  "Share of users with at least two dated reviews whose entire history remains one cluster:", "",
                  "| Dataset | 1 day | 7 days | 30 days | 90 days |", "| --- | ---: | ---: | ---: | ---: |"])
    for r in results:
        values = r["review_clusters_30_days"]["threshold_sensitivity"]
        lines.append("| " + r["dataset"] + " | " + " | ".join(fmt(v["multi_dated_user_one_cluster_share"],True) for v in values) + " |")
    lines.extend(["", "## Findings by dataset", ""])
    for r in results:
        c = r["review_clusters_30_days"]
        lines.append(f"- **{r['dataset']}**: median {fmt(r['distributions']['user_events']['median'])} reviews across "
                     f"{fmt(r['user_span_days']['median'])} observed days; median {fmt(c['cluster_count']['median'])} clusters. "
                     f"{fmt(c['multi_dated_user_one_cluster_share'],True)} of multi-dated users fit in one cluster. "
                     f"{fmt(r['tied_gap_share'],True)} of adjacent dated reviews share a timestamp; "
                     f"{fmt(r['repeated_event_share'],True)} of rows repeat a user-item pair.")
    lines.extend(["", "![Population clustering patterns](cluster_patterns.png)", "",
                  "## User timelines", "",
                  "With `--plots`, `user_review_timelines.png` and `.pdf` show every dataset in two views: calendar dates and days since each selected user's first observed review. Each dataset also has its own `review_timelines.png` and `.pdf`.", "",
                  "Samples are deterministic users at P10, P25, P50, P75, P90, P95, P99, and P100 of event-history length among users with at least two dated events; ties are ordered by user ID. This is a deliberately stratified illustration, not a random population sample. Labels give event counts and 30-day cluster counts. Ticks represent reviews; colored segments connect reviews in the same cluster. Coincident ticks overlap. Full-population statistics above should guide comparisons.", "",
                  "![Chronological user review timelines](user_review_timelines.png)", "",
                  "## Interpretation and limits", "",
                  "- The simultaneous one-pass filter can leave users and items below five interactions. Counts here describe their actual surviving histories.",
                  "- Distinct-item histories can be shorter than event histories. Repeated reviews or duplicated crawls do not establish repeated consumption. Same-pair timestamp excess rows are recorded separately in each JSON summary.",
                  "- Steam dates are day-level, so within-day order and sessions cannot be inferred. Ties also limit sequence order in other datasets; timestamp ties use source row order only as a deterministic convention.",
                  "- Steam identities use the release's username field; these names are not reconciled to stable account IDs.",
                  "- MovieLens records rating entry times; a concentrated rating burst can summarize much earlier movie viewing. Review/rating gaps measure recording activity rather than consumption cadence.",
                  "- Positive-gap item adjacency may depend on tie resolution at either endpoint. It is an audit metric, not a behavioral transition estimate.",
                  "- Filtering removes intermediate events and can enlarge observed gaps. Histories are category-specific and censored at dataset boundaries. Comparisons reflect different collection methods and time windows.",
                  "- Three events suffice for one training event plus one validation and one test event only by row count. Timestamp ties, repeated items, and chronological leakage need separate treatment before any split.", "",
                  "## Files", "",
                  "Each dataset directory contains event/distinct-entity length histograms, length and gap buckets, compressed user/item profiles, and `history_summary.json`. Profile timestamps use Unix seconds (UTC). JSON includes length-cap loss, gap counts, and exact interpolated distribution quantiles.", ""])
    write_csv(output / "comparison.csv", columns, rows)
    write_csv(output / "clustering_comparison.csv",
              ["dataset", "users", "median_events", "median_observed_span_days", "median_clusters_30_days",
               "one_cluster_share_among_users_with_2plus_dated_events", "median_largest_cluster_event_share",
               "median_positive_gap_days", "tied_adjacent_gap_share"],
              ([r["dataset"],r["users"],r["distributions"]["user_events"]["median"],
                r["user_span_days"]["median"],r["review_clusters_30_days"]["cluster_count"]["median"],
                r["review_clusters_30_days"]["multi_dated_user_one_cluster_share"],
                r["review_clusters_30_days"]["largest_cluster_share"]["median"],
                r["positive_gap_days"]["median"],r["tied_gap_share"]] for r in results))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_comparison(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for r in results:
        with (output / r["dataset"] / "user_events.csv").open(encoding="utf-8") as handle:
            hist = [(int(row["length"]), int(row["entities"])) for row in csv.DictReader(handle)]
        if hist:
            lengths, counts = zip(*hist)
            survival = []
            remaining = sum(counts)
            for count in counts:
                survival.append(remaining / r["users"])
                remaining -= count
            axes[0].step(lengths, survival, where="pre", label=r["dataset"])
    axes[0].set(xscale="log", yscale="log", xlabel="History length (events)", ylabel="Fraction of users with at least this length")
    axes[0].legend(fontsize=7)
    labels = [r["dataset"].replace("amazon23_", "") for r in results]
    axes[1].barh(labels, [100*(r["repeated_event_share"] or 0) for r in results])
    axes[1].set(xlabel="Repeated-pair rows (%)")
    axes[2].barh(labels, [100*(r["tied_gap_share"] or 0) for r in results])
    axes[2].set(xlabel="Adjacent dated events with tied timestamps (%)")
    fig.tight_layout()
    fig.savefig(output / "history_patterns.png", dpi=180)
    plt.close(fig)


def plot_timelines(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    def draw(axes, name):
        folder = output / name
        with (folder / "timeline_users.csv").open(encoding="utf-8") as handle:
            users = list(csv.DictReader(handle))
        events = {u["user_id"]:[] for u in users}
        with gzip.open(folder / "timeline_events.csv.gz", "rt", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                events[row["user_id"]].append(float(row["timestamp_epoch_seconds"]))
        labels = []
        for index, user in enumerate(users):
            times = events[user["user_id"]]
            labels.append(f"{user['length_percentile']}  n={user['events']}  c={user['clusters_30_days']}")
            if not times:
                continue
            clusters = [[times[0]]]
            for value in times[1:]:
                if value-clusters[-1][-1]>30*86400:
                    clusters.append([])
                clusters[-1].append(value)
            for j, cluster in enumerate(clusters):
                color = ("#1864ab", "#e67700")[j%2]
                calendar = [mdates.date2num(datetime.fromtimestamp(t,timezone.utc)) for t in cluster]
                relative = [(t-times[0])/86400 for t in cluster]
                for ax,xs in zip(axes,(calendar,relative)):
                    ax.hlines(index,xs[0],xs[-1],color=color,linewidth=2,alpha=.55)
                    ax.scatter(xs,[index]*len(xs),marker="|",s=20,color=color,alpha=.65,linewidths=.8)
        for ax in axes:
            ax.set_yticks(range(len(users)),labels,fontsize=7)
            ax.set_ylim(len(users)-.5,-.5)
            ax.grid(axis="x",alpha=.2)
        locator = mdates.AutoDateLocator(minticks=3,maxticks=7)
        axes[0].xaxis.set_major_locator(locator)
        axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[0].set_title(name+" — calendar dates",fontsize=10)
        axes[1].set_title("Days since first observed review",fontsize=10)
        axes[1].set_xlim(left=-5)

    if not results:
        return
    fig, axes = plt.subplots(len(results),2,figsize=(17,3.5*len(results)),squeeze=False)
    for row,result in zip(axes,results):
        draw(row,result["dataset"])
        single, panels = plt.subplots(1,2,figsize=(17,4.5))
        draw(panels,result["dataset"])
        single.suptitle("Observed review timelines; clusters split at gaps >30 days (n=events, c=clusters)",fontsize=11)
        single.tight_layout()
        for suffix in ("png","pdf"):
            single.savefig(output/result["dataset"]/f"review_timelines.{suffix}",dpi=160)
        plt.close(single)
    fig.suptitle("User review timelines across all datasets\nStratified examples by history-length percentile; ticks = reviews, colors = 30-day-gap clusters",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.975))
    for suffix in ("png","pdf"):
        fig.savefig(output/f"user_review_timelines.{suffix}",dpi=160)
    plt.close(fig)


def plot_clusters(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["dataset"].replace("amazon23_", "") for r in results]
    fig,axes = plt.subplots(1,3,figsize=(16,5))
    values = [([r["review_clusters_30_days"]["cluster_count"]["median"] or 0 for r in results],"Median clusters per dated user"),
              ([100*(r["review_clusters_30_days"]["multi_dated_user_one_cluster_share"] or 0) for r in results],"Multi-dated users in one cluster (%)"),
              ([100*(r["review_clusters_30_days"]["largest_cluster_share"]["median"] or 0) for r in results],"Median share of reviews in largest cluster (%)")]
    for ax,(data,label) in zip(axes,values):
        ax.barh(labels,data,color="#1864ab")
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis="x",alpha=.2)
    fig.suptitle("All retained users: review clusters split at inactivity gaps >30 days")
    fig.tight_layout()
    for suffix in ("png","pdf"):
        fig.savefig(output/f"cluster_patterns.{suffix}",dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/histories"))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    folders = sorted(p.parent for p in args.results.glob("*/summary.json"))
    if args.datasets:
        missing = set(args.datasets) - {p.name for p in folders}
        if missing:
            parser.error(f"missing dataset summaries: {sorted(missing)}")
        folders = [p for p in folders if p.name in args.datasets]
    if not folders:
        parser.error("no dataset results found; first run analyze.py --export")
    for folder in folders:
        if not (folder / "filtered_interactions.csv.gz").is_file():
            parser.error(f"{folder}: missing export; run analyze.py --export")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for folder in folders:
        print(f"Analyzing histories: {folder.name}", flush=True)
        results.append(run_dataset(folder, args.output, args.work_dir))
        write_report(results, args.output)
    if args.plots:
        plot_comparison(results, args.output)
        plot_timelines(results, args.output)
        plot_clusters(results, args.output)
    print(f"History report written to {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
