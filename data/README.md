# Data Download Instructions

All datasets are publicly available. Run `python src/causal_book/data/download.py` to fetch them automatically, or follow the manual instructions below.

---

## Oregon Health Insurance Experiment (OHE)

**Used in**: Chapters 1–14, 21, 31–36

**Source**: Finkelstein et al. (2012). NBER Working Paper 17190.

**Direct download**:
```python
import urllib.request, pathlib

base = "https://data.nber.org/oregon/"
files = [
    "oregonhie_descriptive_vars.dta",
    "oregonhie_survey12m_vars.dta",
    "oregonhie_ed_vars.dta",
    "oregonhie_inperson_vars.dta",
]
out = pathlib.Path("data/raw/ohe")
out.mkdir(parents=True, exist_ok=True)
for f in files:
    urllib.request.urlretrieve(base + f, out / f)
    print(f"Downloaded {f}")
```

**Key variables**:

| Variable | Description |
|----------|-------------|
| `selected` | Lottery selection indicator (instrument Z) |
| `ohp_all_ever_admin` | Ever enrolled in Medicaid (treatment D) |
| `doc_any_12m` | Any doctor visit in past 12 months |
| `catastrophic_exp_inp` | Catastrophic out-of-pocket expenses |
| `health_genflip_bin_12m` | Self-reported health good/excellent |
| `numhh_list` | Household size (stratification variable) |
| `female_inp` | Female indicator |
| `age_19_34_inp`, `age_35_49_inp`, `age_50_64_inp` | Age group indicators |
| `income_cat_det_inp` | Income category |

**Citation**: Finkelstein, A., Taubman, S., Wright, B., Bernstein, M., Gruber, J., Newhouse, J. P., ... & Oregon Health Study Group. (2012). The Oregon health insurance experiment: Evidence from the first year. *Quarterly Journal of Economics*, 127(3), 1057–1106.

---

## BRFSS (Behavioral Risk Factor Surveillance System)

**Used in**: Chapters 15–20 (ACA DiD), 25–30 (G-methods)

**Source**: CDC, annual surveys 2008–2018.

**Download**: Annual XPT files from CDC:
```python
import urllib.request, zipfile, pathlib

years = range(2008, 2019)
out = pathlib.Path("data/raw/brfss")
out.mkdir(parents=True, exist_ok=True)

for year in years:
    url = f"https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"
    dest = out / f"LLCP{year}XPT.zip"
    urllib.request.urlretrieve(url, dest)
    with zipfile.ZipFile(dest) as z:
        z.extractall(out / str(year))
    print(f"Downloaded BRFSS {year}")
```

**Key variables used**:
- `_STATE` — FIPS state code
- `HLTHPLN1` — Health plan coverage (treatment proxy)
- `GENHLTH` — General health (1=excellent … 5=poor)
- `MEDCOST` — Could not see doctor due to cost
- `CHECKUP1` — Routine checkup recency
- `_INCOMG` — Income category

---

## ACA Medicaid Expansion Dates

**Used in**: Chapters 15–20

**Source**: Kaiser Family Foundation (KFF)

Manual download from KFF or use the bundled file:
```
data/raw/aca/medicaid_expansion_dates.csv
```

Columns: `state`, `fips`, `expansion_date` (YYYY-MM-DD), `expanded` (0/1)

---

## Simulated Longitudinal Data

**Used in**: Chapters 25–30 (G-methods)

Generated via `src/causal_book/data/simulate.py`. No download required.

```python
from causal_book.data.simulate import simulate_brfss_longitudinal
A, L, Y, U = simulate_brfss_longitudinal(n=5000, T=4, seed=42)
```
