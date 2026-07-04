# HYROX Results — Data Dictionary

Rows: **1,667,989**  |  Columns: **69**

One row per athlete per event/division/gender. Times are `HH:MM:SS` (or `MM:SS.ss`); every `*_time` has a numeric `*_seconds` companion.

| Column | Meaning |
| --- | --- |
| `season` | HYROX season number (from the season-N results URL) |
| `year` | Calendar year parsed from the event name |
| `city` | Host city / race location |
| `division` | Division + race day (e.g. HYROX PRO - Friday) |
| `event_id` | Site event identifier (division + race slot) |
| `sex` | M, W, or X |
| `idp` | Site athlete/result id (unique within an event) |
| `name` | Athlete name (Last, First) |
| `nationality` | 3-letter nationality code |
| `age_group` | Age class (e.g. 30-34) |
| `rank_overall` | Finishing place within division+gender |
| `rank_age_group` | Finishing place within age group |
| `finish_time` | Net finish time (start mat to finish) |
| `overall_time` | Gross/official overall time |
| `rank_overall_detail` | — |
| `run_1_time` | Run 1 split time |
| `run_2_time` | Run 2 split time |
| `run_3_time` | Run 3 split time |
| `run_4_time` | Run 4 split time |
| `run_5_time` | Run 5 split time |
| `run_6_time` | Run 6 split time |
| `run_7_time` | Run 7 split time |
| `run_8_time` | Run 8 split time |
| `run_total_time` | Run Total split time |
| `best_run_lap_time` | Best Run Lap split time |
| `ski_erg_time` | Ski Erg split time |
| `ski_erg_place` | Place within the Ski Erg split |
| `sled_push_time` | Sled Push split time |
| `sled_push_place` | Place within the Sled Push split |
| `sled_pull_time` | Sled Pull split time |
| `sled_pull_place` | Place within the Sled Pull split |
| `burpee_broad_jump_time` | Burpee Broad Jump split time |
| `burpee_broad_jump_place` | Place within the Burpee Broad Jump split |
| `row_time` | Row split time |
| `row_place` | Place within the Row split |
| `farmers_carry_time` | Farmers Carry split time |
| `farmers_carry_place` | Place within the Farmers Carry split |
| `sandbag_lunges_time` | Sandbag Lunges split time |
| `wall_balls_time` | Wall Balls split time |
| `wall_balls_place` | Place within the Wall Balls split |
| `granular_timing_json` | Full In/Out timing table (time-of-day, cumulative, diff) as JSON |
| `sandbag_lunges_place` | Place within the Sandbag Lunges split |
| `run_total_place` | Place within the Run Total split |
| `best_run_lap_place` | Place within the Best Run Lap split |
| `run_6_place` | Place within the Run 6 split |
| `run_7_place` | Place within the Run 7 split |
| `roxzone_time` | Roxzone split time |
| `roxzone_place` | Place within the Roxzone split |
| `finish_seconds` | Numeric seconds for `finish` / `finish_time` |
| `overall_seconds` | Numeric seconds for `overall` / `overall_time` |
| `run_1_seconds` | Numeric seconds for `run_1` / `run_1_time` |
| `run_2_seconds` | Numeric seconds for `run_2` / `run_2_time` |
| `run_3_seconds` | Numeric seconds for `run_3` / `run_3_time` |
| `run_4_seconds` | Numeric seconds for `run_4` / `run_4_time` |
| `run_5_seconds` | Numeric seconds for `run_5` / `run_5_time` |
| `run_6_seconds` | Numeric seconds for `run_6` / `run_6_time` |
| `run_7_seconds` | Numeric seconds for `run_7` / `run_7_time` |
| `run_8_seconds` | Numeric seconds for `run_8` / `run_8_time` |
| `run_total_seconds` | Numeric seconds for `run_total` / `run_total_time` |
| `best_run_lap_seconds` | Numeric seconds for `best_run_lap` / `best_run_lap_time` |
| `ski_erg_seconds` | Numeric seconds for `ski_erg` / `ski_erg_time` |
| `sled_push_seconds` | Numeric seconds for `sled_push` / `sled_push_time` |
| `sled_pull_seconds` | Numeric seconds for `sled_pull` / `sled_pull_time` |
| `burpee_broad_jump_seconds` | Numeric seconds for `burpee_broad_jump` / `burpee_broad_jump_time` |
| `row_seconds` | Numeric seconds for `row` / `row_time` |
| `farmers_carry_seconds` | Numeric seconds for `farmers_carry` / `farmers_carry_time` |
| `sandbag_lunges_seconds` | Numeric seconds for `sandbag_lunges` / `sandbag_lunges_time` |
| `wall_balls_seconds` | Numeric seconds for `wall_balls` / `wall_balls_time` |
| `roxzone_seconds` | Numeric seconds for `roxzone` / `roxzone_time` |