import numpy as np
import pandas as pd

from kwincome.clean import clean, occupation_major
from kwincome.codes import OCC_MAJOR
from kwincome.load import load_job_codes, load_raw
from kwincome.paths import JOB_XLSX


def test_load_handles_bom_and_blanks(raw_csv, job_table):
    df = load_raw(raw_csv)
    assert "id" in df.columns
    out, _ = clean(df, job_table)
    assert out["occupation_code"].isna().sum() == 2
    assert out["occupation_label"].isna().sum() == 2


def test_labels_mapped(raw_csv, job_table):
    out, _ = clean(load_raw(raw_csv), job_table)
    assert out.loc[1, "gender_label"] == "Female"
    assert out.loc[1, "region_label"] == "Gangwon & Chungbuk"
    assert list(out["education_label"].cat.categories)[0] == "No schooling (<7)"
    assert out["education_label"].cat.ordered


def test_marriage_labels(raw_csv, job_table):
    out, _ = clean(load_raw(raw_csv), job_table)
    assert out.loc[0, "marriage_label"] == "Married"   # code 1
    assert out.loc[1, "marriage_label"] == "Widowed"   # code 2
    assert pd.isna(out.loc[2, "marriage_label"])       # code 9: nonresponse
    assert pd.isna(out.loc[3, "marriage_label"])       # code 0: nonresponse


def test_nonresponse_to_nan(raw_csv, job_table):
    out, _ = clean(load_raw(raw_csv), job_table)
    assert pd.isna(out.loc[2, "reason_label"])
    assert pd.isna(out.loc[2, "marriage_label"])  # marriage 9
    assert out.loc[3, "reason_label"] == "nursing"


def test_no_reason_group_column(raw_csv, job_table):
    out, _ = clean(load_raw(raw_csv), job_table)
    assert "reason_group" not in out.columns


def test_unmapped_occupation(raw_csv, job_table):
    out, rep = clean(load_raw(raw_csv), job_table)
    assert out.loc[1, "occupation_label"] == "Unmapped"
    assert "999" in rep["unmapped_codes"]
    assert rep["occupation_mapping_rate"] < 1
    assert rep["n_rows"] == 4
    assert out.loc[0, "occupation_label"] == "Crop farmers"


def test_occupation_major(raw_csv, job_table):
    out, _ = clean(load_raw(raw_csv), job_table)
    assert out.loc[0, "occupation_major"] == "6"
    assert pd.isna(out.loc[2, "occupation_major"])


def test_occupation_major_military_and_unknown():
    codes = pd.Series(["1011", "1012", "1001", "111", "9999", "999", "91", np.nan], dtype=object)
    major = occupation_major(codes)
    assert list(major.iloc[:4]) == ["A", "A", "A", "1"]   # 4-digit 10xx is armed forces
    assert pd.isna(major.iloc[4])                         # 9999: unknown, not elementary
    assert list(major.iloc[5:7]) == ["9", "9"]
    assert pd.isna(major.iloc[7])
    assert OCC_MAJOR["A"] == "Armed forces"


def test_clean_does_not_mutate_input(raw_csv, job_table):
    df = load_raw(raw_csv)
    before = df.copy()
    clean(df, job_table)
    pd.testing.assert_frame_equal(df, before)


def test_job_codes_real_table():
    jc = load_job_codes(JOB_XLSX)
    assert list(jc.columns) == ["job_code", "job_title"]
    assert jc["job_code"].is_unique and jc["job_code"].notna().all()
