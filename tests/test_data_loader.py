import io
import textwrap

import pandas as pd

from automobile_analysis.data_loader import COLUMN_NAMES, load_data


class TestColumnNames:
    def test_length(self):
        assert len(COLUMN_NAMES) == 26

    def test_first_and_last(self):
        assert COLUMN_NAMES[0] == "symboling"
        assert COLUMN_NAMES[-1] == "price"

    def test_no_duplicates(self):
        assert len(COLUMN_NAMES) == len(set(COLUMN_NAMES))


class TestLoadData:
    def test_loads_from_csv_file(self, tmp_path):
        csv_content = textwrap.dedent("""\
            3,?,alfa-romero,gas,std,two,convertible,rwd,front,88.6,168.8,64.1,48.8,2548,dohc,four,130,mpfi,3.47,2.68,9.0,111,5000,21,27,13495
            1,?,audi,gas,std,four,sedan,fwd,front,99.8,176.6,66.2,54.3,2337,ohc,four,109,mpfi,3.19,3.40,10.0,102,5500,24,30,17450
        """)
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        df = load_data(str(csv_file))

        assert len(df) == 2
        assert list(df.columns) == COLUMN_NAMES
        # '?' should become NaN
        assert pd.isna(df["normalized-losses"].iloc[0])

    def test_question_mark_becomes_nan(self, tmp_path):
        csv_content = "3,?,alfa-romero,gas,std,two,convertible,rwd,front,88.6,168.8,64.1,48.8,2548,dohc,four,130,mpfi,3.47,2.68,9.0,?,5000,21,27,?\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        df = load_data(str(csv_file))
        assert pd.isna(df["normalized-losses"].iloc[0])
        assert pd.isna(df["horsepower"].iloc[0])
        assert pd.isna(df["price"].iloc[0])
