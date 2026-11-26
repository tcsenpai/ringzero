# ringzero demo (M8): a plain stdlib script, run with `!run python3 /demo/stats.py`
import csv
import json
import re
import statistics
from collections import Counter

with open("/demo/people.csv", newline="") as f:
    rows = list(csv.DictReader(f))

ages = [int(r["age"]) for r in rows]
domains = Counter(re.search(r"@([\w.]+)$", r["email"]).group(1) for r in rows)
report = {
    "people": len(rows),
    "mean_age": round(statistics.mean(ages), 1),
    "by_city": dict(Counter(r["city"] for r in rows)),
    "domains": dict(domains.most_common()),
    "oldest": max(rows, key=lambda r: int(r["age"]))["name"],
}
print(json.dumps(report, indent=2, ensure_ascii=False))

with open("/demo/report.json", "w") as f:
    json.dump(report, f)
print("written /demo/report.json")
