"""Run the full analysis suite, then render every output table into tables.html."""

import csv
import html
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAB = HERE / "output" / "tables"

SCRIPTS = [
    "01_participation.py",
    "02_finish_times.py",
    "03_stations.py",
    "04_pacing.py",
    "05_performance_drivers.py",
    "06_elites.py",
    "07_athletes.py",
    "08_deep_dive.py",
]

SECTIONS = {
    "01": "Participation & Demographics",
    "02": "Finish Times",
    "03": "Stations",
    "04": "Pacing & Fatigue",
    "05": "Performance Drivers",
    "06": "Elites & Records",
    "07": "Athlete Careers",
    "08": "Deep Dives",
}

TIME_COL = re.compile(r"(seconds|^p\d+$|^med$|^q\d+$|^win$|median_winning_time|^first_time$)")
MAX_ROWS = 60


def fmt_hms(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_cell(col: str, val: str) -> str:
    if val == "":
        return "–"
    try:
        f = float(val)
    except ValueError:
        return html.escape(val)
    if TIME_COL.search(col) and f > 120:
        return fmt_hms(f)
    if f == int(f):
        return f"{int(f):,}"
    return f"{f:,.2f}"


def render_table(path: Path) -> str:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    truncated = len(body) > MAX_ROWS
    body = body[:MAX_ROWS]
    thead = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    trs = []
    for r in body:
        tds = "".join(f"<td>{fmt_cell(header[i] if i < len(header) else '', v)}</td>"
                      for i, v in enumerate(r))
        trs.append(f"<tr>{tds}</tr>")
    note = (f'<p class="note">Showing first {MAX_ROWS} rows — full data in '
            f'<code>output/tables/{path.name}</code></p>' if truncated else "")
    title = path.stem.replace("_", " ").title()
    return (f'<div class="tbl-card" id="{path.stem}">'
            f"<h3>{html.escape(title)}</h3>"
            f'<div class="tbl-scroll"><table><thead><tr>{thead}</tr></thead>'
            f"<tbody>{''.join(trs)}</tbody></table></div>{note}</div>")


CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,'Helvetica Neue',Arial,sans-serif;
     background:#f8fafc;color:#1e293b;padding:40px 24px;max-width:1200px;margin:0 auto}
h1{font-size:26px;margin-bottom:4px}
.sub{color:#64748b;margin-bottom:28px;font-size:14px}
h2{font-size:19px;margin:36px 0 12px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}
h3{font-size:14px;margin-bottom:8px;color:#0f172a}
.tbl-card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
          padding:18px 18px 12px;margin-bottom:18px;box-shadow:0 1px 3px rgba(15,23,42,.05)}
.tbl-scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{background:#f1f5f9;color:#334155;text-align:left;padding:7px 10px;white-space:nowrap;
   position:sticky;top:0}
td{padding:6px 10px;border-top:1px solid #f1f5f9;white-space:nowrap;
   font-variant-numeric:tabular-nums}
tr:hover td{background:#f8fafc}
.note{font-size:11.5px;color:#94a3b8;margin-top:8px}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:11px}
a.back{display:inline-block;margin-bottom:18px;color:#0284c7;text-decoration:none;font-size:13px}
"""


def build_tables_page() -> None:
    csvs = sorted(TAB.glob("*.csv"))
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
             f"<title>Hyrox Analysis — Data Tables</title><style>{CSS}</style></head><body>",
             '<a class="back" href="index.html">← back to dashboard</a>',
             "<h1>Hyrox Analysis — Data Tables</h1>",
             f'<p class="sub">{len(csvs)} tables generated from 1.18M unique race results · '
             "times shown as h:mm:ss</p>"]
    current = None
    for p in csvs:
        prefix = p.stem[:2]
        if prefix != current and prefix in SECTIONS:
            parts.append(f"<h2>{prefix} · {SECTIONS[prefix]}</h2>")
            current = prefix
        parts.append(render_table(p))
    parts.append("</body></html>")
    out = HERE / "tables.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"tables page: {out}")


def main() -> None:
    for script in SCRIPTS:
        print(f"Running {script}...")
        result = subprocess.run([sys.executable, str(HERE / script)],
                                capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            sys.exit(1)
    build_tables_page()
    print("All analysis scripts completed successfully.")


if __name__ == "__main__":
    main()
