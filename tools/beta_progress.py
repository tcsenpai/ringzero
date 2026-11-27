#!/usr/bin/env python3
"""Progress bar of the beta gate, read live from the mycelium db.

The beta task (T34, "Beta release gate") and the tasks it directly depends on are counted;
open ones are listed in dependency order, the first one is the one in progress.
  python3 tools/beta_progress.py            # print once
  python3 tools/beta_progress.py --watch    # refresh every 20 s (Ctrl-C to stop)
"""
import os, sqlite3, sys, time

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".mycelium", "mycelium.db")
BETA_TITLE = "Beta release gate"
WIDTH = 30


def snapshot():
    # read-only, so it never competes with myc writes
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    tasks = {i: (t, s) for i, t, s in con.execute("select id, title, status from tasks")}
    deps = {}
    for t, d in con.execute("select task_id, depends_on_task_id from dependencies"):
        deps.setdefault(t, set()).add(d)
    con.close()
    beta = next(i for i, (t, _) in tasks.items() if t == BETA_TITLE)
    items = sorted(deps.get(beta, set())) + [beta]

    # dependency order inside the beta set: a task comes after what it depends on
    order, seen = [], set()
    def visit(i):
        if i in seen:
            return
        seen.add(i)
        for d in sorted(deps.get(i, set()) & set(items)):
            visit(d)
        order.append(i)
    for i in items:
        visit(i)
    return tasks, order


def render():
    tasks, order = snapshot()
    done = [i for i in order if tasks[i][1] == "closed"]
    todo = [i for i in order if tasks[i][1] != "closed"]
    frac = len(done) / len(order)
    fill = round(frac * WIDTH)
    out = [f"ringzero beta  [{'#' * fill}{'.' * (WIDTH - fill)}]  {len(done)}/{len(order)}  {frac:.0%}", ""]
    for i in order:
        title, status = tasks[i]
        if status == "closed":
            mark = "done"
        elif todo and i == todo[0]:
            mark = "NOW "
        else:
            mark = "    "
        out.append(f"  {mark}  T{i:<3} {title}")
    out.append("")
    out.append(time.strftime("  updated %H:%M:%S"))
    return "\n".join(out)


if __name__ == "__main__":
    if "--watch" in sys.argv:
        try:
            while True:
                sys.stdout.write("\033[2J\033[H" + render() + "\n")
                sys.stdout.flush()
                time.sleep(20)
        except KeyboardInterrupt:
            pass
    else:
        print(render())
