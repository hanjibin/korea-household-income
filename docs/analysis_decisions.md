# Analysis decisions

Every judgment call made while rebuilding the analysis, with the alternatives considered and what the choice does to the results. Numbers come from the executed notebook ([`notebooks/analysis.ipynb`](../notebooks/analysis.ipynb)); section numbers refer to it. "Main specification" means: outcome log real equivalized income, data years (Kaggle `year`) 2008-2017, income > 0, full controls, n = 66,449 household-years (8,814 households), controlled gap -7.3% (95% CI -10.1% to -4.5%), explained share 88% of the 0.621 log-point gap.

## Data meaning

| Decision | Alternatives | Why | Effect on results |
|---|---|---|---|
| Income unit read as 10,000 KRW per year (estimated) | 1,000 KRW per year; 10,000 KRW per month | No codebook ships with the data. Only the first reading gives a plausible annual household income (median about 30 million KRW in 2018); the others are implausibly low or high (Section 2) | Labels only. Log gaps, percent gaps and decomposition shares do not depend on the unit. Every income figure is marked "estimated" |
| Income year = Kaggle `year` (inference: `year` appears to be the income reference year, fielded the following year); deflated with that year's CPI | Income year = `year` - 1 (the earlier assumption: `year` as the fieldwork year) | Evidence (inference, from the source survey's sample design as recalled, not from a codebook): Kaggle year 2005 = wave 1 with exactly 7,072 households, and the survey's wave 1 (7,072 households) was fielded in 2006 and asked about 2005 income; the refresh sample (wave 7) was fielded in 2012, and 1,878 new ids appear in Kaggle year 2011 | Controlled gap and Oaxaca split unaffected because the models include year fixed effects (controlled gap -7.3% and unexplained 0.076 in both readings). Levels and the trend shift by one CPI year: raw log gap 0.622 -> 0.621, explained 0.546 -> 0.545, trend now income years 2005-2018 with +58% (was 2004-2017, +56%), regional medians lower (Seoul 2,342 -> 2,288) |
| Compare female-headed and male-headed households, not women and men | Treat `gender` as the individual's gender | `income` is household income and person attributes describe the head. 1,428 ids change head gender or birth year across waves (1,284 change head gender), for example after a husband's death | Changes the question: every result is a household comparison. Wording rule: "female-headed households", never "women earn" |
| Marital status codes labelled 1 Married, 2 Widowed, 3 Divorced, 4 Separated, 5 Never married, 6 Other; 0 and 9 as missing | Leave as "code k"; confirm from a codebook | No labels in the data. The profile fits the source survey's usual convention: code 2 is 89% female-headed, median head age 74, median household size 1; code 1 is 3% female-headed, size 3; code 5 median head age 36. Codes 3 and 4 rest on convention only | Labels only; numbers unchanged if relabelled. Makes the 0.068 log-point marital contribution interpretable. 68% of female-headed household-years are code 2 (Widowed, inferred), computed on all years |
| Non-working heads enter the models by raw reason code ("Not employed: code k"), without labels | Group by the labels in the Kaggle description (as the original course notebook did) | The documented labels conflict with the data: code 4, documented as preparing for school, has 704 rows, 95% female-headed, median head age 68, 78% widowed; only code 3 (n = 52, median age 27.5) looks like students (Section 9) | Models do not depend on any label grouping. The reason-group chart was replaced by a per-code profile table, reported as a data-quality finding; no result is interpreted by reason |
| Occupation used only for data years (Kaggle `year`) 2008-2017 (`occ_consistent`) | Use all years; recode across classifications | Unmapped occupation codes are 44.0%, 43.7% and 45.1% of coded rows in 2005-2007, 0.0% in 2008-2017 (2 rows in 2008) and 3.4% in 2018. The occupation mix changes 6.1% between 2007 and 2008 vs 0.7% to 2.3% in other years. Most consistent with a classification change (inference) | Models with occupation use 10 of 14 data years. Robustness on all years 2005-2018 without occupation terms (n = 92,625): controlled gap -7.2% (CI -9.8% to -4.4%), explained share 88%. That row differs from the main one in both sample and controls |
| Major occupation group = first digit of the code, except 4-digit codes starting with 10 (1011 military officer, 1012 long-term sergeant) -> "Armed forces" (A) and 9999 -> missing | First digit for every code | The first digit put military codes into Managers (about 200 rows in 2008-2017) and the unknown code 9999 (16 rows, 2 of them in 2008-2017; no title in the table) into Elementary occupations | Armed forces appears as its own group in the occupation chart; the 9999 rows enter models as a missing occupation group. Headline shares unchanged at the reported precision (work/occupation 0.181, 29%) |
| Province-to-region grouping for the map (for example Busan and Ulsan with Gyeongnam, Incheon with Gyeonggi, Sejong with Chungnam) | No map; table only | The survey codes 7 regions without listing provinces; the grouping is an assumption (`kwincome.geo`) | Affects the map drawing only. Models use the survey's region codes directly; region and year together contribute about 0 log points to the explained gap |
| No survey weights | Weighted estimates | The Kaggle extract has no weight column (checked in Section 2); the source survey is reported to oversample low-income households | Results describe this sample, not Korea. Not quantifiable here |

## Measurement

| Decision | Alternatives | Why | Effect on results |
|---|---|---|---|
| Main measure: equivalized income = income / sqrt(household size) | Household income; per-person income (income / size, the original course metric) | The square-root scale (OECD) is the common middle ground between no and full economies of scale. Female-headed households average 1.50 people vs 2.90, and 68% live alone, so the choice moves the gap | Median gap: household income -68%, equivalized -52%, per person -29% (Section 5). Controlled gap with log household income instead: -8.2% (CI -11.0% to -5.4%), explained share 91%, vs -7.3% and 88% in the main specification |
| Deflate to 2018 prices with World Bank `FP.CPI.TOTL` (Korea), rebased 2018 = 100, CPI of the income year | Nominal income; another CPI series | Nominal values mix inflation with income growth over 14 years | Real equivalized median +58% from income year 2005 to 2018. Controlled gap unaffected because year fixed effects absorb year-level deflation |
| Same outcome (log real equivalized income) for the raw step and every model | Different outcomes per analysis | The first staged step then equals the raw log gap (-0.621) and the decomposition total | The 52% median gap (all years) and the 46% log gap (2008-2017) differ by design; both are reported and labelled |

## Cleaning and outliers

| Decision | Alternatives | Why | Effect on results |
|---|---|---|---|
| Keep income <= 0 rows; exclude them only from log models, with a count | Drop everywhere; log(income + c) | Medians handle them; a log needs positive values and an arbitrary shift would distort the bottom of the distribution | 232 rows <= 0 in all data; 111 dropped from the main model sample |
| Medians for summaries; winsorizing at the 1st/99th percentile as a sensitivity, not the main run | Means; trimming; winsorize the main run | The maximum income is far above the median; medians and logs already limit outlier influence | Winsorized outcome: controlled gap -7.5% (CI -10.2% to -4.8%) vs -7.3% |
| Suppress groups with fewer than 100 household-years | Show all groups; merge into "Other" | Tiny groups give extreme, unstable medians | 73 detailed occupations remain (2008-2017, mapped codes); no headline number depends on a suppressed group |
| Unmapped occupation codes labelled "Unmapped", with a mapping rate | Silent NaN | Makes coverage visible | 89.8% of coded rows map to a job title; unmapped rows sit almost entirely outside 2008-2017 |
| No rows dropped in cleaning; blanks and nonresponse codes (for example 99, 9) become missing | Drop incomplete rows | Keeps the sample visible and comparable across sections | 92,857 rows after cleaning |

## Inference

| Decision | Alternatives | Why | Effect on results |
|---|---|---|---|
| Household-cluster bootstrap for medians (resample whole ids), 1,000 draws, seed 42 | Row-level bootstrap; analytic CI | Households repeat up to 14 times and heads can change gender within an id; resampling rows would understate uncertainty | Trend and group charts show these CIs. The row-level alternative was not computed |
| OLS standard errors clustered by household (`id`) | Conventional SEs | Same repeated-household structure | Main controlled gap CI -10.1% to -4.5% |
| Oaxaca CIs from a household-cluster bootstrap, 500 draws (100 in the robustness table, where CIs are not reported) | Analytic delta-method SEs | Consistent with the clustering elsewhere | Unexplained 0.076 (95% CI 0.045 to 0.107); explained 0.545 (0.515 to 0.575) |
| Quantile regression CIs left unclustered (statsmodels default), stated as a limitation | Household-cluster bootstrap of each quantile | A clustered bootstrap of QuantReg costs many minutes of runtime; conclusions rest on the clustered OLS and Oaxaca CIs | Quantile CIs are likely too narrow. Controlled gap 6.3% to 11.2% across conditional quantiles, widest at the 90th; treated as suggestive only |

## Decomposition

| Decision | Alternatives | Why | Effect on results |
|---|---|---|---|
| Headline: pooled two-fold Oaxaca-Blinder (Neumark / Jann) with a female-head dummy in the pooled regression | Male coefficients; female coefficients; three-fold with interaction | Standard choice that avoids picking one group as the norm; no interaction term, so explained and unexplained shares sum to 100% | Explained 0.545 (87.7%), unexplained 0.076 (12.3%). Male reference: explained 90.7%, unexplained 0.058. Female reference: 84.8%, unexplained 0.094. Range 85% to 91% |
| Decomposition as the headline, staged regressions as the narrative | Staged regressions only | Staged step sizes depend on entry order (age entered first takes credit that education might otherwise get); the decomposition does not | Work/occupation 0.181 (29%) and education 0.171 (28%) are the largest pieces; age, education and work/occupation together are 0.427 (69%), the rest of the explained part is household makeup and marital status. Staged steps are shown but not used for attribution |
| Report contributions in log points and shares of the log gap; percent income differences only as a separate, labelled quantity | Mix shares and percent gaps | The unexplained 0.076 log points is 12% of the log gap but corresponds to a 7.3% lower income after controls; side by side they invite misreading | Copy never pairs "88%" with "7%" as if both were shares |
| Read "explained" as composition, not causes | Interpret each contribution causally | Marital status and household size (0.119 log points, 19% of the log gap together) largely restate what female headship means here (widowed heads, often living alone) | Stated in the notebook, README and limitations |

## Presentation

| Decision | Alternatives | Why | Effect on results |
|---|---|---|---|
| Static charts carry the story; ipywidgets only in a final "Explore it yourself" section | Widgets throughout | Widgets do not render on GitHub or nbviewer; the story must be complete without a kernel | None on numbers. The folium map is also interactive-only, with a static PNG of the same data |
| Every number in the notebook text is computed; a check cell (Section 12) asserts the quoted numbers | Hand-typed numbers | Prevents drift between text and results | Notebook execution fails if a quoted number changes |
