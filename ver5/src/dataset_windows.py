"""Short-agent defaults from the plotted 30-day temporal-group statistics.

Source: preliminary_grouped/outputs/additional_dated_20261023/<dataset>/status.json
Cohort: sampled test users, observed history capped at 100 interactions.
Yelp statistics use the dated matched subsets. Preference history stays unchanged.
"""
from decimal import Decimal, ROUND_HALF_UP

MEAN_INTERACTIONS_PER_GROUP = {'Steam': 2.2513798294029104, 'MovieLens1M': 39.475130890052355, 'Yelp19': 3.302574411516763, 'Yelp23': 2.86813122437024}
ALIASES = {
    "steam": "Steam", "movielens": "MovieLens1M", "movielens1m": "MovieLens1M",
    "movielens-1m": "MovieLens1M", "ml-1m": "MovieLens1M",
    "yelp19": "Yelp19", "yelp23": "Yelp23",
}


def window_argument(value):
    """Allow an explicit positive integer or dataset-based automatic selection."""
    return None if str(value).lower() == "auto" else int(value)


def default_short_window(dataset, fallback=10):
    name = ALIASES.get(dataset.lower())
    if name is None:
        return fallback
    mean = Decimal(str(MEAN_INTERACTIONS_PER_GROUP[name]))
    return int((mean * 2).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
