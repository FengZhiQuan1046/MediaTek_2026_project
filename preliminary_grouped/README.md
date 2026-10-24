# Dataset structure and input mappings

The pipeline treats all datasets as a user–item interaction graph, with optional rating and timestamp. It reads interaction files; companion metadata files are described here to explain the source structure. Source versions may have different counts, so statistics are computed from your files rather than copied from a published benchmark.

## Expected local layout

```text
data/
  amazon23/
    All_Beauty.jsonl.gz
    Baby_Products.jsonl
    Sports_and_Outdoors.jsonl
    Toys_and_Games.jsonl
  yelp/
    yelp_academic_dataset_review.json
  ml-1m/
    ratings.dat
  steam/
    steam_reviews.json.gz
```

JSON/JSONL inputs contain **one object per line**, not one large JSON array. Plain text and gzip (`.gz`) are supported. Extract ZIP/TAR archives first. Inputs use UTF-8; only MovieLens `ratings.dat` is read, so the different encoding sometimes used for movie titles does not affect ingestion.

The current configuration uses UCSD's compressed All Beauty file and the lab's official Hugging Face mirror for the other three Amazon categories. The mirror serves uncompressed JSONL files. Steam can be obtained from the original UCSD host or its checksum-verified community mirror using `download_datasets.py`; the local Steam filename and format are identical.

## Amazon Reviews 2023

Use the raw review files for `All_Beauty`, `Baby_Products`, `Sports_and_Outdoors`, and `Toys_and_Games`. These are separate datasets in the configuration. Review records map as follows:

| Field | Meaning / handling |
| --- | --- |
| `user_id` | Reviewer ID → normalized user |
| `parent_asin` | Parent product ID → normalized item |
| `asin` | Variant ID; unused |
| `rating` | Numeric rating, 1–5 |
| `timestamp` | Unix milliseconds → UTC |

Additional review fields include `title`, `text`, `images`, `verified_purchase`, and `helpful_vote`. Product metadata joins through `parent_asin`; companion records include product descriptions, categories, pricing, and media. Variants can share a parent, which affects repeated-pair counts.

The raw Baby Products file contains at least one `rating: 0.0` record (line 2,903,667 in the downloaded release), despite the documented 1–5 range. The analyzer retains these user–item interactions, converts out-of-range star ratings to null, and reports their counts and original numeric values in `summary.json`. Raw files remain unchanged.

The official benchmark scripts remove duplicate pairs and apply iterative k-core filtering. This tool starts with raw reviews and uses the single-pass rule in the README. [Official schema and downloads](https://amazon-reviews-2023.github.io/main.html); [benchmark preprocessing](https://github.com/hyp1231/AmazonReviews2023/blob/main/benchmark_scripts/README.md).

Synthetic review shape:

```json
{"user_id":"user-A","parent_asin":"parent-1","asin":"variant-1","rating":4.0,"timestamp":1609459200000,"text":"Example review"}
```

For official rating-only **raw/0-core** CSV exports, switch the entry to `"format": "csv"` and set `"item_field": "parent_asin", "timestamp_unit": "ms"`. CSV inputs require a header. Do not concatenate overlapping raw and processed releases.

## Yelp Open Dataset

Use `yelp_academic_dataset_review.json`, a JSON-lines file. A review links a user to a business:

| Field | Meaning / handling |
| --- | --- |
| `user_id` | User ID → normalized user |
| `business_id` | Business ID → normalized item |
| `stars` | Numeric rating, 1–5 |
| `date` | Date/time string → UTC |
| `review_id` | Review ID; unused |
| `text`, `useful`, `funny`, `cool` | Review text and feedback; unused |

Companion business and user records join by their respective IDs. Business data includes location, categories, attributes, hours, and aggregate rating information. Other files include tips and check-ins; they are different event types and are not mixed with reviews here. Aggregate `review_count` values are not used to compute filter eligibility. Naive review date strings are treated as UTC consistently; this does not recover a business's local timezone.

[Official dataset](https://www.yelp.com/dataset); [official documentation](https://www.yelp.com/dataset/documentation/main); [Yelp's archived examples](https://github.com/YelpArchive/dataset-examples). The documentation endpoint blocked automated access during implementation; review fields should be checked against the downloaded release if using a historical variant.

## MovieLens 1M

MovieLens 1M uses headerless text files separated by `::`:

| File | Field order |
| --- | --- |
| `ratings.dat` | `UserID::MovieID::Rating::Timestamp` |
| `users.dat` | `UserID::Gender::Age::Occupation::Zip-code` |
| `movies.dat` | `MovieID::Title::Genres` |

Only ratings are required. User/movie IDs become strings, ratings are 1–5, and timestamps are Unix **seconds**. Companion files join on user/movie IDs. Age and occupation are encoded categories; genres are separated by `|`. The release contains 1,000,209 ratings and 6,040 users, with each original user having at least 20 ratings. Item filtering can still affect the resulting graph.

[GroupLens release and download](https://grouplens.org/datasets/movielens/1m/); [official README/schema](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt).

## Steam: choose the release explicitly

The default configuration uses McAuley/UCSD **version 2 review data**, commonly named `steam_reviews.json.gz`. Other Steam datasets have different interaction meanings and must use the matching adapter. JSON objects and Python dictionary literals are supported for Steam files; literals are parsed with `ast.literal_eval`, never executed.

| Configuration format | Source structure | Interaction |
| --- | --- | --- |
| `steam_reviews` (default) | Flat records: `username`, `product_id`, `date`, plus review text and other attributes | One review; no numeric rating by default |
| `steam_user_reviews` | Australian/version 1 records: outer `user_id`, nested `reviews` list with `item_id`, `recommend`, `posted`, `review` | One nested review; recommendation becomes 0/1 |
| `steam_items` | Australian/version 1 records: outer `user_id`, nested `items` list with `item_id`, `item_name`, `playtime_forever`, `playtime_2weeks` | One owned game, including zero-playtime games; no rating or date |
| `csv` | Your headered normalized export | One row under your chosen event definition |

Version 2 `date` strings are parsed as ISO dates. Version 1 `posted` values such as `Posted January 1, 2015.` are parsed as dates. Dates omitting the year become null; the pipeline does not infer a year. Playtime is not used as a rating. Steam metadata, e.g. game `id`, title, tags, and genres, is a separate item table and is not an interaction input.

To use the Australian reviews, replace the Steam entry:

```json
{"name":"steam","format":"steam_user_reviews","paths":["data/steam/australian_user_reviews.json.gz"]}
```

For ownership instead, use `"format": "steam_items"` with `australian_users_items.json.gz`. These yield different graphs; use separate dataset names if analyzing both.

For another flat release, field mappings can be overridden:

```json
{"name":"steam","format":"steam_reviews","paths":["data/steam/reviews.jsonl.gz"],"user_field":"user_id","item_field":"app_id","timestamp_field":"timestamp","timestamp_unit":"s","rating_field":null}
```

Supported timestamp units are `s`, `ms`, `iso`, and `posted`. Optional field names can be null to disable those fields. Field mappings refer to top-level keys after nested reviews/items are flattened. IDs must be nonempty strings or integers; stable Steam IDs are preferable when provided by your release. The default version 2 adapter uses its supplied `username` identity without attempting account reconciliation.

[UCSD Steam releases and citations](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data). The schema above distinguishes the two review releases and ownership data; inspect your own release before choosing an adapter.

## Common normalized structure

| Column | Type | Meaning |
| --- | --- | --- |
| `user_id` | String | Original user identity within this dataset |
| `item_id` | String | Parent product, business, movie, or game identity |
| `rating` | Nullable numeric | Stars or recommendation; empty when unavailable |
| `timestamp_utc` | Nullable ISO string | UTC event date/time; original precision retained up to microseconds |

SQLite internally calls the last column `timestamp`. The optional compressed CSV export uses `timestamp_utc`; set `"timestamp_field": "timestamp_utc"` when reading it with the generic CSV adapter. A review with a missing date still counts as an interaction. IDs are not remapped into consecutive integers, and no metadata-only users/items are counted.
