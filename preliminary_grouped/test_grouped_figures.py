import unittest
from plot_grouped_preliminary import group_lags, gpu_summary
from amazon_group_lengths import DAY_MS


class GroupedFiguresTests(unittest.TestCase):
    def test_recent_first_groups_preserve_exact_30day_boundary_and_ties(self):
        dates=[0,0,DAY_MS,31*DAY_MS,62*DAY_MS]
        self.assertEqual(group_lags(dates),{1:1,2:2,3:2,4:2,5:2})

    def test_missing_timestamp_is_not_silently_assigned_to_a_group(self):
        with self.assertRaises(ValueError):group_lags([0,None])

    def test_user_bootstrap_constant_values_and_singleton(self):
        summary=gpu_summary([2,2,2],25252,20,'cpu')
        self.assertEqual(summary['mean'],2)
        self.assertEqual(summary['ci_low'],2)
        self.assertEqual(summary['ci_high'],2)
        self.assertEqual(summary['n_users'],3)
        self.assertIsNone(gpu_summary([1],25252,20,'cpu')['ci_low'])
