"""Region geometry: province boundaries dissolved into the survey's 7 regions."""
import geopandas as gpd
import pandas as pd

from .paths import GEOJSON

REGION_COL = "region_label"

# Keys are the `shapeName` property of data/external/kr_provinces.geojson
# (geoBoundaries KOR ADM1). The grouping into the survey's 7 regions is an
# assumption: the survey documentation does not list which provinces each region
# code contains. Sejong is in the file and is grouped with Chungnam.
PROVINCE_TO_REGION = {
    "Seoul": "Seoul",
    "Gyeonggi": "Gyeonggi", "Incheon": "Gyeonggi",
    "South Gyeongsang": "Gyeongnam", "Busan": "Gyeongnam", "Ulsan": "Gyeongnam",
    "North Gyeongsang": "Gyeongbuk", "Daegu": "Gyeongbuk",
    "South Chungcheong": "Chungnam", "Daejeon": "Chungnam", "Sejong": "Chungnam",
    "Gangwon": "Gangwon & Chungbuk", "North Chungcheong": "Gangwon & Chungbuk",
    "North Jeolla": "Jeolla & Jeju", "South Jeolla": "Jeolla & Jeju",
    "Gwangju": "Jeolla & Jeju", "Jeju": "Jeolla & Jeju",
}


def region_geodata(values: pd.DataFrame, value_col: str) -> gpd.GeoDataFrame:
    """Dissolve provinces into the 7 regions and attach `values[value_col]`.

    `values` needs a `region_label` column (one row per region). Regions without
    a value get NaN.
    """
    provinces = gpd.read_file(GEOJSON)
    unmapped = set(provinces["shapeName"]) - set(PROVINCE_TO_REGION)
    if unmapped:
        raise ValueError(f"provinces without a region: {sorted(unmapped)}")
    provinces[REGION_COL] = provinces["shapeName"].map(PROVINCE_TO_REGION)
    regions = provinces[[REGION_COL, "geometry"]].dissolve(by=REGION_COL, as_index=False)
    return regions.merge(values[[REGION_COL, value_col]], on=REGION_COL, how="left")
