import pandas as pd
import pytest

from dataset.processed.preprocess_module import add_rul, load_data


def test_add_rul_clips_and_computes_correctly():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "op_setting_1": [0, 0, 0, 0, 0],
            "op_setting_2": [0, 0, 0, 0, 0],
            "op_setting_3": [0, 0, 0, 0, 0],
            **{f"sensor_{i}": [0, 0, 0, 0, 0] for i in range(1, 22)},
        }
    )

    out = add_rul(df)

    assert "RUL" in out.columns
    assert out.loc[(out["unit"] == 1) & (out["cycle"] == 1), "RUL"].iloc[0] == 2
    assert out.loc[(out["unit"] == 2) & (out["cycle"] == 2), "RUL"].iloc[0] == 0


def test_load_data_parses_whitespace_file(tmp_path):
    row = [1, 1] + [0.1, 0.2, 0.3] + [float(i) for i in range(1, 22)]
    text = "   ".join(str(v) for v in row) + "\n"
    file_path = tmp_path / "sample.txt"
    file_path.write_text(text, encoding="utf-8")

    df = load_data(str(file_path))

    assert df.shape == (1, 26)
    assert list(df.columns[:2]) == ["unit", "cycle"]


def test_load_data_raises_on_unexpected_columns(tmp_path):
    bad_row = [1, 1, 0.1]
    file_path = tmp_path / "bad.txt"
    file_path.write_text(" ".join(str(v) for v in bad_row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_data(str(file_path))
