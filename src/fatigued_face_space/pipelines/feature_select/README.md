# feature_select

The pipeline aggregates per-subject medians from `features_table`, compares
fatigue groups with the baseline, and summarizes direction agreement.

Parameters select subjects and fatigue classes. All feature columns in
`features_table` are included automatically, in input column order. Metadata
columns (`Name`, `person`, `check_date`, `class`, `fatigue_level`, `is_baseface`)
and CSV-generated `Unnamed:` index columns are excluded. New numeric feature
columns require no parameter changes.
All configured subjects remain in each median table, including subjects with
missing groups. Missing observations within a group are excluded from its
median; an absent/all-missing group yields `missing`.

Three median tables are saved as Parquet under `data/06_fatigue_group`, and two
comparison tables as Parquet under `data/07_compared_table`.
`fatigue_direction_matches` is saved as CSV under
`data/08_fatigue_direction_matches`. All six datasets are versioned using Kedro's
standard layout: `<dataset filename>/<version>/<dataset filename>`.
The final table has rows `languid` and `tense` and one column per feature.
Its row index is saved. Plot datasets are saved under `data/plots`.

Summary precedence: any missing comparison yields `missing`; uniform directions
yield `up-match`, `down-match`, or `unchanged-match`; any other mixture containing
`unchanged` yields `unchanged-mixed`; a mixture of only `up` and `down` yields
`nomatch`.

Run synthetic-data pipeline and edge-case checks:

```powershell
.venv\Scripts\python.exe -m pytest tests/pipelines/feature_select -q --no-cov -o pythonpath=src
```
