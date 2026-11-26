# External data sources

## cpi_kor.csv
- Indicator: Consumer price index (2010 = 100), `FP.CPI.TOTL`, Korea, Rep.
- Source: World Bank, World Development Indicators
- URL: https://api.worldbank.org/v2/country/KOR/indicator/FP.CPI.TOTL?format=json&date=2004:2018
- Access date: 2026-11-26
- License: CC BY 4.0
- Stored as `year,cpi` (original base 2010 = 100); `load_cpi` rebases to 2018 = 100.

## kr_provinces.geojson
- Content: South Korea first-level administrative boundaries (17 provinces and metropolitan cities, Sejong included), property `shapeName` is the English name.
- Source: geoBoundaries gbOpen, KOR ADM1, simplified geometry (189 KB, no further simplification needed). Underlying boundary source: Natural Earth.
- URL: https://www.geoboundaries.org/data/snapshots/1c423398df24ddf6e730b116567d2181c9782778d0459c3f3b76ca2f9f12b6bc/gbOpen/KOR/ADM1/geoBoundaries-KOR-ADM1_simplified.geojson (metadata: https://www.geoboundaries.org/api/current/gbOpen/KOR/ADM1/)
- Access date: 2026-11-26
- License: Public Domain, as reported by the geoBoundaries API (`boundaryLicense`; licence source https://www.naturalearthdata.com/about/terms-of-use/). geoBoundaries asks for citation: Runfola et al. (2020), PLoS ONE 15(4): e0231866.
- Why not `southkorea/southkorea-maps`: the GitHub repository has no licence file (GitHub reports no licence), so reuse terms are unclear.
- Use: provinces are grouped into the 7 survey regions in `kwincome.geo.PROVINCE_TO_REGION` (grouping is an assumption, see that module).
