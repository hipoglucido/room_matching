import unittest
import pandas as pd

from rooms.data_processing import split_data


class TestSplitData(unittest.TestCase):

    def test_split_data_valid_percentages(self):
        # Create a sample DataFrame
        df = pd.DataFrame({"A": range(100), "target_variable": range(100)})

        # Split the data
        train_df, val_df, test_df = split_data(df, train_pct=0.7, val_pct=0.15)
        tolerance = 5
        # Check the sizes of the splits
        self.assertAlmostEqual(len(train_df), 70, delta=tolerance)
        self.assertAlmostEqual(len(val_df), 15, delta=tolerance)
        self.assertAlmostEqual(len(test_df), 15, delta=tolerance)

    def test_split_data_invalid_percentages(self):
        # Create a sample DataFrame
        df = pd.DataFrame({"A": range(100), "target_variable": range(100)})

        # Test with invalid percentages (sum greater than 1.0)
        with self.assertRaises(ValueError):
            split_data(df, train_pct=0.8, val_pct=0.3)

    def test_split_data_edge_case(self):
        # Create a sample DataFrame
        df = pd.DataFrame({"A": range(100), "B": range(100)})

        # Test with edge case (train_pct + val_pct = 1.0)
        with self.assertRaises(ValueError):
            split_data(df, train_pct=0.9, val_pct=0.1)
