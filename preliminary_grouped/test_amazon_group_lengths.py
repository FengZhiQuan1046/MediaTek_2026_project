from datetime import datetime, timezone
import unittest
from amazon_group_lengths import DAY_MS, summarize, time_groups


def ms(text):
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()*1000)


class GroupLengthTests(unittest.TestCase):
    def test_matches_supplied_30day_fixture_and_retains_ties(self):
        dates=[ms(d) for d in ['2020-01-01','2020-01-01','2020-01-02','2020-02-01','2020-03-03']]
        groups=time_groups([dates[-1],None,*dates[:-1]])
        self.assertEqual(list(map(len,groups)),[4,1])
        self.assertEqual((groups[0][-1]-groups[0][0])/DAY_MS,31)

    def test_group_weighting_and_latest_group_index(self):
        a=[ms(d) for d in ['2020-01-01','2020-01-01','2020-01-02','2020-02-01','2020-03-03']]
        result=summarize({0:a,1:[ms('2020-02-01')],2:[None,None]},[0,1,2])
        self.assertEqual(result['groups'],3)
        self.assertEqual(result['dated_interactions'],6)
        self.assertEqual(result['missing_timestamp_interactions'],2)
        self.assertEqual(result['mean_interactions_per_group'],2)
        self.assertEqual(result['user_weighted_mean_interactions_per_group'],1.75)
        self.assertEqual(result['by_group_from_most_recent'][0]['mean_interactions'],1)
        self.assertEqual(result['by_group_from_most_recent'][1]['mean_interactions'],4)

    def test_empty_groups_are_not_zero_length_observations(self):
        result=summarize({0:[None]},[0])
        self.assertEqual(result['groups'],0)
        self.assertIsNone(result['mean_interactions_per_group'])
