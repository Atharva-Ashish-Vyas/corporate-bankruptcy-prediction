# Dataset provenance

**Dataset:** Polish Companies Bankruptcy, Sebastian Tomczak, 2016.

**DOI:** https://doi.org/10.24432/C5F600  
**Official page:** https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data  
**License:** Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/

The unmodified `raw/3year.arff` is bundled for offline reproduction. It contains 10,503 financial statements, 64 numeric financial indicators, and a binary target. The label refers to bankruptcy within the three-year forecasting horizon. Missing values use `?`.

SHA-256: `662617d2f90bfd17223595e50a0ac7007f9ee7915cfb519dee0422b31dd13ebc`

The general UCI page describes a combined representation with a year field and 65 features. This project loads the individual third-year ARFF file, which has **64 predictors plus one target**, with no year feature.

The preprocessing removes 87 exact duplicate rows, including two positive rows. It retains 10,416 unique statements and 493 positive labels. No identical feature vectors with conflicting labels were found. Detailed counts and the raw-file checksum appear in `reports/data_audit.json`.

No company identifier is provided. Removing exact duplicates does not establish that all statements from the same company occupy only one partition. The reported test is a stratified random holdout, not a chronological or verified company-disjoint test.

To recover a missing copy: run `python scripts/download_data.py`. This fetches the official archive, extracts only `3year.arff`, and verifies its checksum.
