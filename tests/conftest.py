import pandas as pd
import pytest

HEADER = ("id,year,wave,region,income,family_member,gender,year_born,"
          "education_level,marriage,religion,occupation,company_size,reason_none_worker")
ROWS = [
    "1,2010,6,1,500,2,1,1960,5,1,1,611,3, ",
    "1,2011,7,6,0,2,2,1960,7,2,9,999,4, ",
    "2,2010,6,6,300,1,2,1950,3,9,1, , ,99",
    "3,2010,6,2,100,3,2,1970,4,0,2, , ,8",
]


@pytest.fixture
def raw_csv(tmp_path):
    p = tmp_path / "raw.csv"
    p.write_bytes(("\n".join([HEADER, *ROWS]) + "\n").encode("utf-8-sig"))
    return p


@pytest.fixture
def job_table():
    return pd.DataFrame({"job_code": ["611", "111"], "job_title": ["Crop farmers", "Officials"]})
