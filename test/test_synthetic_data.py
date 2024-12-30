from rooms.synthetic_data import generate_synthetic_dataset
import pytest

def test_generate_synthetic_dataset():
    n_rows = 10000
    match_ratio = .2
    df = generate_synthetic_dataset(n_rows, match_ratio)
    assert len(df) == n_rows
    actual = float(df['match'].mean())
    assert actual == pytest.approx(match_ratio, .1)


