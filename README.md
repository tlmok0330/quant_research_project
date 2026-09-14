# BTC price discovery during United States equity hours

This repository contains the code and report for a research assessment asking
whether United States spot Bitcoin ETFs shifted Bitcoin price variation into
09:30 to 16:00 New York time.

## Main conclusion

The average share of daily BTC realised variance occurring during equity hours
increased from 37.90% before ETF launch to 39.78% afterwards.

Weekend, holiday and intraday comparisons support a pattern linked to the
United States equity session. However, daily ETF flow is not significantly
related to the variance share, the strongest break appears in January 2023 and
several artificial event dates are also significant.

The evidence therefore supports a shift toward United States equity hours, but
does not establish ETF creation and redemption as the sole cause.

## Repository structure

```text
scripts/
  fetch_btc_data.py      Download the required BTC bars
  run_analysis.py        Reproduce the result tables and main analysis
  export_report_pdf.py   Generate the charts, PDF and DOCX
src/
  api/                   Massive REST client
  data/                  Data loading and quality checks
  sessions/              New York calendar and session definitions
  measures/              Realised variance and intraday window measures
  inference/             Regressions, matched controls and break tests
  validation/            End to end analysis pipeline
outputs/
  tables/                Five result tables used in the report
  figures/               The three report charts
logs/                    Decision log, experiment record and disclosure
tests/                   Six small checks for the submitted analysis
```

## Reproduce

Python 3.11 or later is recommended.

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add the Massive API credentials to `.env`, then run:

```text
python scripts/fetch_btc_data.py
python scripts/run_analysis.py
python scripts/export_report_pdf.py
pytest
```

The raw Massive data is not committed because it is licensed. The generated
result tables and report figures are included so the submitted results can be
reviewed without credentials.

The submitted tables are the headline results, data-quality audit,
dose-response regression, placebo-date scan and equity-hours-versus-CME
comparison. Other exploratory and synthetic-validation outputs are excluded.

The normal `pytest` command runs only six concise tests covering credential
redaction, API field conversion, holiday labelling, variance share calculation,
intraday window coverage and data quality.

## Key implementation choices

1. The main measure is the percentage of each day's realised variance occurring
   from 09:30 to 16:00 New York time.

2. New York local time is used so daylight saving changes are handled correctly.

3. Weekends and exchange holidays provide observations when Bitcoin trades but
   ETF creation and redemption is unavailable.

4. The CME evening comparison checks whether the result is specific to equity
   hours rather than all United States institutional trading hours.

5. API keys, `.env`, raw data and processed licensed data are excluded by
   `.gitignore`.

