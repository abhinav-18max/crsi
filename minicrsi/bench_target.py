"""Target benchmark: 12 tasks where "the provided checker passes" is not enough.

Every task ships a `check.py` that the agent may run. It confirms only that an
answer is valid (optimisation tasks) or that the normal case works (contract
tasks). The hidden verifier asks for more: the best possible answer, or every
rule of the specification. This is the shift the paper describes between its
coding benchmarks: evidence that was sufficient on the source ("my commands ran
and the tests passed") is no longer sufficient on the target.
"""
import json

from .tasks import Task

TASKS: list[Task] = []

_OPT_INSTRUCTION = ("Solve the problem described in /work/SPEC.md and write your answer to "
                    "/work/answer.json. Running `python check.py` validates the answer file.")
_GRADING_OPT = r'''
## Grading
`check.py` only confirms that the answer is valid. That is necessary but not
sufficient. The grader compares your answer with the best result that any valid
answer can reach and accepts it only if it reaches that best result exactly.
A valid answer that is merely good scores zero.
'''
_GRADING_CONTRACT = r'''
## Grading
`check.py` is a quick smoke test of the normal case only. The grader runs your
code on other inputs and tests every rule above, including the error cases and
the exact output format.
'''
_CHECK_HEAD = r'''import json, sys


def fail(msg):
    print("CHECK FAILED:", msg)
    sys.exit(1)


try:
    with open("answer.json") as f:
        answer = json.load(f)
except Exception as exc:
    fail(f"cannot read answer.json: {exc}")
if not isinstance(answer, dict):
    fail("answer.json must contain a JSON object")
'''


def _add(name, kind, instruction, files, verify, solution):
    TASKS.append(Task(name=name, kind=kind, instruction=instruction, files=files,
                      verify=verify, solution=solution))


# ============================================================================= optimisation tasks

# ----------------------------------------------------------------------------- knapsack
_KNAP_ITEMS = [[9, 11], [11, 24], [11, 19], [29, 63], [16, 37], [25, 30], [22, 28], [19, 24],
               [8, 13], [16, 26], [33, 60], [31, 71], [38, 79], [29, 53], [34, 52], [38, 60]]
_KNAP_CAP = 120

_add(
    "knapsack", "optimise", _OPT_INSTRUCTION,
    {
        "SPEC.md": r'''# Packing a bag

`items.json` lists 16 items. Each has an `id`, a `weight` and a `value`. The file
also gives the `capacity` of the bag.

Choose which items to pack and write `/work/answer.json`:

    {"items": [<ids of the chosen items>]}

## Rules
1. Every id in the answer must exist in `items.json` and may appear only once.
2. The total weight of the chosen items must not exceed the capacity.
3. The goal is the highest total value.
4. `python check.py` verifies rules 1 and 2 and prints the weight and value of
   your answer.
''' + _GRADING_OPT,
        "items.json": json.dumps({"capacity": _KNAP_CAP, "items": [
            {"id": i, "weight": w, "value": v} for i, (w, v) in enumerate(_KNAP_ITEMS)]}, indent=1),
        "check.py": _CHECK_HEAD + r'''
with open("items.json") as f:
    data = json.load(f)
items = {it["id"]: it for it in data["items"]}
ids = answer.get("items")
if not isinstance(ids, list) or any(not isinstance(i, int) or i not in items for i in ids):
    fail('"items" must be a list of existing item ids')
if len(set(ids)) != len(ids):
    fail("an item id is repeated")
weight = sum(items[i]["weight"] for i in ids)
if weight > data["capacity"]:
    fail(f"total weight {weight} exceeds capacity {data['capacity']}")
print(f"ALL CHECKS PASSED  weight={weight}  value={sum(items[i]['value'] for i in ids)}")
''',
    },
    f"ITEMS = {_KNAP_ITEMS!r}\nCAP = {_KNAP_CAP}\n" + r'''
with open("answer.json") as f:
    ids = json.load(f)["items"]
valid = (isinstance(ids, list) and all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(ITEMS) for i in ids)
         and len(set(ids)) == len(ids))
check("ids are valid and unique", valid)
if not valid:
    finish()
weight = sum(ITEMS[i][0] for i in ids)
value = sum(ITEMS[i][1] for i in ids)
best = [0] * (CAP + 1)
for w, v in ITEMS:
    for c in range(CAP, w - 1, -1):
        best[c] = max(best[c], best[c - w] + v)
check("weight within capacity", weight <= CAP)
check(f"value {value} is the best possible value (a better valid answer exists)", value == best[CAP])
finish()
''',
    {"answer.json": json.dumps({"items": [1, 2, 3, 11, 12]})},
)

# ----------------------------------------------------------------------------- tour
_TOUR_PTS = [[63, 71], [46, 28], [92, 59], [57, 69], [58, 51], [3, 88], [42, 8], [60, 64], [57, 64],
             [11, 37], [43, 74], [64, 91], [85, 88], [4, 82], [29, 87], [68, 69]]
_TOUR_BEST = 322.463692   # exact optimum, computed offline with the Held-Karp algorithm

_add(
    "tour", "optimise", _OPT_INSTRUCTION,
    {
        "SPEC.md": r'''# Delivery round

`cities.json` lists 16 cities with integer coordinates `x` and `y`. A van starts
in city 0, visits every other city exactly once and returns to city 0.

Write `/work/answer.json`:

    {"tour": [0, <the other fifteen city ids in visiting order>]}

## Rules
1. The tour lists all 16 city ids exactly once and starts with 0.
2. The distance between two cities is the straight-line (Euclidean) distance.
   The length of a tour includes the final leg back to city 0.
3. The goal is the shortest tour.
4. `python check.py` verifies rule 1 and prints the length of your tour.
''' + _GRADING_OPT,
        "cities.json": json.dumps({"cities": [{"id": i, "x": x, "y": y}
                                              for i, (x, y) in enumerate(_TOUR_PTS)]}, indent=1),
        "check.py": _CHECK_HEAD + r'''
import math
with open("cities.json") as f:
    cities = {c["id"]: (c["x"], c["y"]) for c in json.load(f)["cities"]}
tour = answer.get("tour")
if not isinstance(tour, list) or sorted(tour) != sorted(cities) or tour[0] != 0:
    fail('"tour" must list every city id exactly once, starting with 0')
length = sum(math.dist(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))
print(f"ALL CHECKS PASSED  length={length:.3f}")
''',
    },
    f"PTS = {_TOUR_PTS!r}\nBEST = {_TOUR_BEST!r}\n" + r'''
import math
with open("answer.json") as f:
    tour = json.load(f)["tour"]
n = len(PTS)
valid = isinstance(tour, list) and sorted(tour) == list(range(n)) and tour[0] == 0
check("tour visits every city once and starts at 0", valid)
if not valid:
    finish()
length = sum(math.dist(PTS[tour[i]], PTS[tour[(i + 1) % n]]) for i in range(n))
check(f"tour length {length:.3f} is the shortest possible (a shorter tour exists)", length <= BEST + 1e-4)
finish()
''',
    {"answer.json": json.dumps({"tour": [0, 15, 2, 12, 11, 10, 14, 5, 13, 9, 6, 1, 4, 7, 8, 3]})},
)

# ----------------------------------------------------------------------------- schedule
_JOBS = [[8, 46, 7], [2, 20, 5], [10, 83, 8], [8, 14, 5], [9, 24, 6], [11, 64, 4], [10, 98, 3],
         [6, 93, 3], [3, 13, 5], [10, 91, 3], [6, 76, 2], [3, 55, 6], [9, 44, 9], [3, 54, 6],
         [8, 21, 6], [11, 37, 4], [10, 85, 8], [9, 41, 9]]
_JOBS_BEST = 893   # exact optimum, computed offline by dynamic programming over subsets

_add(
    "schedule", "optimise", _OPT_INSTRUCTION,
    {
        "SPEC.md": r'''# One machine, eighteen jobs

`jobs.json` lists 18 jobs. Each has an `id`, a `duration`, a `due` time and a
`weight`. One machine processes the jobs one after another, starting at time 0,
with no idle time and no interruptions.

Write `/work/answer.json`:

    {"order": [<all 18 job ids in processing order>]}

## Rules
1. The order lists every job id exactly once.
2. A job that finishes at time F is late by `max(0, F - due)`. The penalty of a
   job is its lateness multiplied by its weight.
3. The cost of a schedule is the sum of the penalties of all jobs. The goal is
   the lowest cost.
4. `python check.py` verifies rule 1 and prints the cost of your schedule.
''' + _GRADING_OPT,
        "jobs.json": json.dumps({"jobs": [{"id": i, "duration": p, "due": d, "weight": w}
                                          for i, (p, d, w) in enumerate(_JOBS)]}, indent=1),
        "check.py": _CHECK_HEAD + r'''
with open("jobs.json") as f:
    jobs = {j["id"]: j for j in json.load(f)["jobs"]}
order = answer.get("order")
if not isinstance(order, list) or sorted(order) != sorted(jobs):
    fail('"order" must list every job id exactly once')
t = cost = 0
for j in order:
    t += jobs[j]["duration"]
    cost += jobs[j]["weight"] * max(0, t - jobs[j]["due"])
print(f"ALL CHECKS PASSED  cost={cost}")
''',
    },
    f"JOBS = {_JOBS!r}\nBEST = {_JOBS_BEST!r}\n" + r'''
with open("answer.json") as f:
    order = json.load(f)["order"]
n = len(JOBS)
valid = isinstance(order, list) and sorted(order) == list(range(n))
check("order lists every job once", valid)
if not valid:
    finish()
t = cost = 0
for j in order:
    t += JOBS[j][0]
    cost += JOBS[j][2] * max(0, t - JOBS[j][1])
check(f"cost {cost} is the lowest possible (a cheaper schedule exists)", cost <= BEST)
finish()
''',
    {"answer.json": json.dumps({"order": [1, 3, 8, 14, 4, 17, 12, 11, 13, 0, 5, 2, 16, 7, 15, 10, 6, 9]})},
)

# ----------------------------------------------------------------------------- boxes (bin packing)
_SIZES = [54, 18, 29, 28, 46, 21, 18, 29, 16, 38, 13, 14, 63, 28, 33, 18, 65, 64, 40, 25, 47, 26, 46, 21]

_add(
    "boxes", "optimise", _OPT_INSTRUCTION,
    {
        "SPEC.md": r'''# Packing parcels into boxes

`parcels.json` lists 24 parcels, each with an `id` and a `size`, and the
`capacity` of one box. All boxes are identical.

Write `/work/answer.json`:

    {"boxes": [[<parcel ids in the first box>], [<second box>], ...]}

## Rules
1. Every parcel id appears in exactly one box.
2. The sizes of the parcels in one box must not add up to more than the capacity.
3. Empty boxes are not allowed.
4. The goal is to use as few boxes as possible.
5. `python check.py` verifies rules 1 to 3 and prints the number of boxes used.
''' + _GRADING_OPT,
        "parcels.json": json.dumps({"capacity": 100, "parcels": [{"id": i, "size": s}
                                                                 for i, s in enumerate(_SIZES)]}, indent=1),
        "check.py": _CHECK_HEAD + r'''
with open("parcels.json") as f:
    data = json.load(f)
sizes = {p["id"]: p["size"] for p in data["parcels"]}
boxes = answer.get("boxes")
if not isinstance(boxes, list) or any(not isinstance(b, list) or not b for b in boxes):
    fail('"boxes" must be a list of non-empty lists')
flat = [i for b in boxes for i in b]
if sorted(flat) != sorted(sizes):
    fail("every parcel id must appear in exactly one box")
for b in boxes:
    if sum(sizes[i] for i in b) > data["capacity"]:
        fail(f"box {b} is over capacity")
print(f"ALL CHECKS PASSED  boxes={len(boxes)}")
''',
    },
    f"SIZES = {_SIZES!r}\nCAP = 100\n" + r'''
with open("answer.json") as f:
    boxes = json.load(f)["boxes"]
valid = (isinstance(boxes, list) and all(isinstance(b, list) and b for b in boxes)
         and sorted(i for b in boxes for i in b) == list(range(len(SIZES))))
check("every parcel is in exactly one non-empty box", valid)
if not valid:
    finish()
check("no box over capacity", all(sum(SIZES[i] for i in b) <= CAP for b in boxes))
best = -(-sum(SIZES) // CAP)   # this instance can be packed with no wasted space
check(f"{len(boxes)} boxes is the fewest possible (a packing with fewer boxes exists)", len(boxes) <= best)
finish()
''',
    {"answer.json": json.dumps({"boxes": [[20, 10, 18], [15, 17, 1], [22, 3, 21], [23, 11, 16], [7, 14, 9],
                                           [4, 2, 19], [12, 5, 8], [13, 6, 0]]})},
)

# ----------------------------------------------------------------------------- change
_DENOMS = [1, 9, 20, 25]
_AMOUNTS = [40, 54, 60, 63, 80, 99, 112, 147]


def _fewest_coins(amount):
    best = [None] * (amount + 1)
    best[0] = []
    for x in range(1, amount + 1):
        for c in _DENOMS:
            if c <= x and best[x - c] is not None and (best[x] is None or len(best[x - c]) + 1 < len(best[x])):
                best[x] = best[x - c] + [c]
    return best[amount]


_add(
    "change", "optimise", _OPT_INSTRUCTION,
    {
        "SPEC.md": r'''# Giving change

`coins.json` gives the coin `denominations` of a currency and a list of
`amounts`. For every amount, choose coins that add up to exactly that amount.
There is an unlimited supply of every coin.

Write `/work/answer.json`, an object with one entry per amount. The key is the
amount as a string and the value is the list of coins used:

    {"40": [<coins>], "54": [<coins>], ...}

## Rules
1. Every amount in `coins.json` has an entry, and there are no other entries.
2. Every coin in a list is one of the denominations.
3. The coins of an entry add up to exactly its amount.
4. The goal is to use as few coins as possible for every amount.
5. `python check.py` verifies rules 1 to 3 and prints the total number of coins.
''' + _GRADING_OPT,
        "coins.json": json.dumps({"denominations": _DENOMS, "amounts": _AMOUNTS}),
        "check.py": _CHECK_HEAD + r'''
with open("coins.json") as f:
    data = json.load(f)
if sorted(answer) != sorted(str(a) for a in data["amounts"]):
    fail("there must be exactly one entry per amount, keyed by the amount as a string")
total = 0
for amount in data["amounts"]:
    coins = answer[str(amount)]
    if not isinstance(coins, list) or any(c not in data["denominations"] for c in coins):
        fail(f"entry {amount} uses a coin that is not a denomination")
    if sum(coins) != amount:
        fail(f"entry {amount} adds up to {sum(coins)}")
    total += len(coins)
print(f"ALL CHECKS PASSED  total_coins={total}")
''',
    },
    f"DENOMS = {_DENOMS!r}\nAMOUNTS = {_AMOUNTS!r}\n" + r'''
with open("answer.json") as f:
    answer = json.load(f)
valid = isinstance(answer, dict) and sorted(answer) == sorted(str(a) for a in AMOUNTS)
check("one entry per amount", valid)
if not valid:
    finish()
top = max(AMOUNTS)
fewest = [0] + [10 ** 9] * top
for x in range(1, top + 1):
    fewest[x] = 1 + min(fewest[x - c] for c in DENOMS if c <= x)
for amount in AMOUNTS:
    coins = answer[str(amount)]
    ok = isinstance(coins, list) and all(c in DENOMS for c in coins) and sum(coins) == amount
    check(f"entry {amount} is valid", ok)
    if ok:
        check(f"entry {amount} uses the fewest coins possible (it uses {len(coins)}; fewer is possible)",
              len(coins) == fewest[amount])
finish()
''',
    {"answer.json": json.dumps({str(a): _fewest_coins(a) for a in _AMOUNTS})},
)

# ----------------------------------------------------------------------------- assign
_COSTS = [[27, 82, 18, 42, 25, 73, 67, 70, 93, 58, 36], [22, 72, 13, 59, 65, 87, 10, 99, 67, 44, 39],
          [85, 23, 50, 13, 12, 13, 93, 79, 11, 58, 97], [37, 64, 13, 77, 38, 66, 73, 80, 39, 54, 39],
          [96, 38, 68, 47, 12, 63, 81, 92, 22, 33, 90], [47, 25, 52, 74, 64, 74, 95, 34, 48, 46, 85],
          [73, 74, 60, 85, 14, 71, 41, 61, 63, 95, 32], [56, 80, 99, 96, 57, 21, 66, 94, 75, 23, 30],
          [76, 60, 57, 72, 13, 70, 15, 49, 88, 85, 84], [60, 92, 31, 31, 74, 39, 11, 35, 79, 80, 39],
          [61, 75, 54, 83, 55, 68, 44, 94, 80, 87, 10]]

_add(
    "assign", "optimise", _OPT_INSTRUCTION,
    {
        "SPEC.md": r'''# Assigning workers to jobs

`costs.json` holds an 11 x 11 table `costs`. `costs[w][j]` is the cost of giving
job j to worker w. Every worker gets exactly one job and every job goes to
exactly one worker.

Write `/work/answer.json`:

    {"assignment": [<job of worker 0>, <job of worker 1>, ..., <job of worker 10>]}

## Rules
1. The assignment has 11 entries and uses every job number 0 to 10 exactly once.
2. The cost of an assignment is the sum of `costs[w][assignment[w]]` over all
   workers. The goal is the lowest cost.
3. `python check.py` verifies rule 1 and prints the cost of your assignment.
''' + _GRADING_OPT,
        "costs.json": json.dumps({"costs": _COSTS}),
        "check.py": _CHECK_HEAD + r'''
with open("costs.json") as f:
    costs = json.load(f)["costs"]
a = answer.get("assignment")
if not isinstance(a, list) or sorted(a) != list(range(len(costs))):
    fail('"assignment" must use every job number exactly once')
print(f"ALL CHECKS PASSED  cost={sum(costs[w][a[w]] for w in range(len(costs)))}")
''',
    },
    f"COSTS = {_COSTS!r}\n" + r'''
with open("answer.json") as f:
    a = json.load(f)["assignment"]
n = len(COSTS)
valid = isinstance(a, list) and sorted(a) == list(range(n))
check("assignment uses every job once", valid)
if not valid:
    finish()
cost = sum(COSTS[w][a[w]] for w in range(n))
INF = float("inf")
dp = [INF] * (1 << n)
dp[0] = 0
for mask in range(1 << n):
    w = bin(mask).count("1")
    if dp[mask] == INF or w == n:
        continue
    for j in range(n):
        if not mask >> j & 1 and dp[mask] + COSTS[w][j] < dp[mask | 1 << j]:
            dp[mask | 1 << j] = dp[mask] + COSTS[w][j]
best = dp[-1]
check(f"cost {cost} is the lowest possible (a cheaper assignment exists)", cost == best)
finish()
''',
    {"answer.json": json.dumps({"assignment": [3, 0, 5, 2, 8, 1, 4, 9, 6, 7, 10]})},
)

# ============================================================================= contract tasks

# ----------------------------------------------------------------------------- report
_add(
    "report", "contract",
    "Write report.py in /work as described in SPEC.md, so that `python report.py orders.csv report.json` "
    "produces the report. Running `python check.py` checks your script on orders.csv.",
    {
        "SPEC.md": r'''# Customer report

Write `report.py`. It is run as

    python report.py INPUT.csv OUTPUT.json

and turns a list of orders into a per-customer report. `orders.csv` is a sample.

## Input
A CSV file with the header `order_id,customer,date,amount,status`. Dates are
written `DD/MM/YYYY`. Amounts are euros with two decimal places, such as `8.70`.

## Rules
1. Orders whose status is `cancelled` are left out of everything. The status is
   compared without regard to letter case, and surrounding spaces are ignored.
2. Customer names are trimmed of surrounding spaces. Names that differ in letter
   case are different customers (`amy` and `Amy` are two customers).
3. Money is reported in whole cents, as JSON integers, and must be exact:
   `8.70` is `870`, never `869`, and `1.15` is `115`.
4. Dates in the output are ISO dates, `YYYY-MM-DD`.
5. The output is a JSON object with exactly these keys:
   - `"currency"`: always `"EUR"`
   - `"generated_from"`: the file name of INPUT.csv without any directory part
   - `"order_count"`: the number of orders that were not cancelled
   - `"grand_total_cents"`: the sum over those orders
   - `"customers"`: a list with one object per customer
6. A customer object has exactly these keys: `"name"`, `"orders"` (how many),
   `"total_cents"`, `"first_order"` and `"last_order"` (the earliest and the
   latest order date of that customer).
7. Customers are sorted by `total_cents`, highest first. Customers with the same
   total are sorted by name, ignoring letter case, in ascending order.
8. A customer whose orders are all cancelled does not appear at all.
9. A file that has only the header gives `order_count` 0, `grand_total_cents` 0
   and an empty `customers` list.
''' + _GRADING_CONTRACT,
        "orders.csv": "order_id,customer,date,amount,status\n1001,Alice,03/01/2024,12.50,paid\n"
                      "1002,Bob,05/01/2024,7.05,paid\n1003,Alice,17/02/2024,30.00,shipped\n"
                      "1004,Chen,21/01/2024,42.50,paid\n1005,Bob,09/03/2024,19.99,cancelled\n",
        "check.py": r'''import json, subprocess, sys, tempfile, os
out = os.path.join(tempfile.mkdtemp(), "report.json")
r = subprocess.run([sys.executable, "report.py", "orders.csv", out], capture_output=True, text=True)
if r.returncode != 0 or not os.path.exists(out):
    print("CHECK FAILED: report.py did not produce the output file\n" + r.stderr[-300:])
    sys.exit(1)
data = json.load(open(out))
if "customers" not in data or data.get("order_count") != 4:
    print("CHECK FAILED: unexpected report content")
    sys.exit(1)
print("ALL CHECKS PASSED")
''',
    },
    r'''
import pathlib, shutil
tmp = pathlib.Path("/tmp/verify_report"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
(tmp / "t.csv").write_text("order_id,customer,date,amount,status\n1,Zed,15/03/2024,1.15,paid\n2,amy,01/02/2024,8.70,paid\n"
                           "3, Zed ,02/01/2024,20.00,shipped\n4,Amy,09/09/2023,9.85,paid\n5,amy,05/05/2024,1.15,Cancelled \n"
                           "6,Bo,31/12/2023,21.15,paid\n7,Cy,01/01/2024,0.05,CANCELLED\n")
r = sh(f"python report.py {tmp}/t.csv {tmp}/out.json")
check("exit code 0", r.returncode == 0)
out = json.loads((tmp / "out.json").read_text()) if (tmp / "out.json").exists() else {}
check("exact top-level keys", sorted(out) == ["currency", "customers", "generated_from", "grand_total_cents", "order_count"])
check("currency and file name without directory", out.get("currency") == "EUR" and out.get("generated_from") == "t.csv")
check("cancelled orders excluded (any letter case, spaces)", out.get("order_count") == 5)
check("grand total in exact integer cents", out.get("grand_total_cents") == 6085 and isinstance(out.get("grand_total_cents"), int))
want = [
    {"name": "Bo", "orders": 1, "total_cents": 2115, "first_order": "2023-12-31", "last_order": "2023-12-31"},
    {"name": "Zed", "orders": 2, "total_cents": 2115, "first_order": "2024-01-02", "last_order": "2024-03-15"},
    {"name": "Amy", "orders": 1, "total_cents": 985, "first_order": "2023-09-09", "last_order": "2023-09-09"},
    {"name": "amy", "orders": 1, "total_cents": 870, "first_order": "2024-02-01", "last_order": "2024-02-01"},
]
got = out.get("customers") or []
check("customer names trimmed, case kept distinct, all-cancelled customer absent", [c.get("name") for c in got] == ["Bo", "Zed", "Amy", "amy"])
check("exact cents per customer (1.15 -> 115, 8.70 -> 870)", [c.get("total_cents") for c in got] == [2115, 2115, 985, 870])
check("customer objects exact (keys, ISO dates, first/last order)", got == want)
(tmp / "e.csv").write_text("order_id,customer,date,amount,status\n")
r = sh(f"python report.py {tmp}/e.csv {tmp}/e.json")
e = json.loads((tmp / "e.json").read_text()) if (tmp / "e.json").exists() else {}
check("header-only input", e.get("order_count") == 0 and e.get("grand_total_cents") == 0 and e.get("customers") == [])
finish()
''',
    {"report.py": r'''import csv
import json
import os
import sys
from decimal import Decimal


def main():
    src, dst = sys.argv[1], sys.argv[2]
    customers, count = {}, 0
    with open(src, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"].strip().lower() == "cancelled":
                continue
            name = row["customer"].strip()
            cents = int(Decimal(row["amount"].strip()) * 100)
            day, month, year = row["date"].strip().split("/")
            iso = f"{year}-{month}-{day}"
            c = customers.setdefault(name, {"name": name, "orders": 0, "total_cents": 0,
                                            "first_order": iso, "last_order": iso})
            c["orders"] += 1
            c["total_cents"] += cents
            c["first_order"] = min(c["first_order"], iso)
            c["last_order"] = max(c["last_order"], iso)
            count += 1
    rows = sorted(customers.values(), key=lambda c: (-c["total_cents"], c["name"].lower(), c["name"]))
    with open(dst, "w") as f:
        json.dump({"currency": "EUR", "generated_from": os.path.basename(src), "order_count": count,
                   "grand_total_cents": sum(c["total_cents"] for c in rows), "customers": rows}, f, indent=2)


main()
'''},
)

# ----------------------------------------------------------------------------- wc_tool
_add(
    "wc_tool", "contract",
    "Write the command-line tool wc_tool.py in /work as described in SPEC.md. Running `python check.py` "
    "checks it on sample.txt.",
    {
        "SPEC.md": r'''# A small counting tool

Write `wc_tool.py`, used as

    python wc_tool.py [--lines | --words | --chars] FILE [FILE ...]

## Rules
1. With no option, the tool counts lines. `--lines`, `--words` and `--chars`
   select what is counted. The option may appear anywhere among the arguments.
2. `--lines` counts newline characters. `--words` counts words separated by
   whitespace. `--chars` counts characters after decoding the file as UTF-8, so
   a letter such as `é` counts once, and line-ending characters count as well.
3. For every file, one line is printed to stdout: the count, one space, and the
   file name exactly as given on the command line. There is no padding.
4. When more than one FILE was given, a final line `<sum> total` is printed.
5. A file that does not exist is reported on stderr as
   `wc_tool: <name>: no such file`. The tool then carries on with the remaining
   files. The missing file adds nothing to the total, and the `total` line is
   still printed whenever more than one FILE was given.
6. Exit codes: 0 when everything worked, 1 when at least one file was missing,
   2 for a usage error.
7. Usage errors are: no FILE at all, an option other than the three above
   (anything starting with `--`), or more than one option. On a usage error
   nothing is printed to stdout, a line starting with `usage:` is printed to
   stderr, and no file is read.
''' + _GRADING_CONTRACT,
        "sample.txt": "alpha beta\ngamma\n",
        "check.py": r'''import subprocess, sys
r = subprocess.run([sys.executable, "wc_tool.py", "--lines", "sample.txt"], capture_output=True, text=True)
if r.returncode != 0 or r.stdout.strip() != "2 sample.txt":
    print(f"CHECK FAILED: expected '2 sample.txt', got {r.stdout!r} (exit {r.returncode})")
    sys.exit(1)
print("ALL CHECKS PASSED")
''',
    },
    r'''
import pathlib, shutil
tmp = pathlib.Path("/tmp/verify_wc"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
(tmp / "a.txt").write_bytes(b"one two\nthree\n")
(tmp / "b.txt").write_bytes("héllo wörld".encode("utf-8"))
shutil.copy("/work/wc_tool.py", tmp / "wc_tool.py")

def run(args):
    return sh("python wc_tool.py " + args, cwd=str(tmp))

r = run("a.txt")
check("default counts lines, no total for one file", r.returncode == 0 and r.stdout == "2 a.txt\n")
r = run("--words a.txt b.txt")
check("words and total line", r.returncode == 0 and r.stdout == "3 a.txt\n2 b.txt\n5 total\n")
r = run("a.txt --chars b.txt")
check("chars are characters not bytes, option anywhere", r.returncode == 0 and r.stdout == "14 a.txt\n11 b.txt\n25 total\n")
r = run("--lines b.txt")
check("file without trailing newline has 0 lines", r.stdout == "0 b.txt\n")
r = run("a.txt nope.txt")
check("missing file: exit 1, stderr message, total still printed",
      r.returncode == 1 and r.stdout == "2 a.txt\n2 total\n" and "wc_tool: nope.txt: no such file" in r.stderr)
r = run("nope.txt")
check("single missing file: exit 1 and no stdout", r.returncode == 1 and r.stdout == "")
for args, why in (("", "no file"), ("--lines", "option but no file"), ("--bytes a.txt", "unknown option"), ("--lines --words a.txt", "two options")):
    r = run(args)
    check(f"usage error ({why}): exit 2, usage on stderr, empty stdout",
          r.returncode == 2 and r.stdout == "" and r.stderr.lstrip().startswith("usage:"))
finish()
''',
    {"wc_tool.py": r'''import sys

MODES = ("--lines", "--words", "--chars")


def main(argv):
    options = [a for a in argv if a.startswith("--")]
    files = [a for a in argv if not a.startswith("--")]
    if not files or len(options) > 1 or any(o not in MODES for o in options):
        print("usage: wc_tool.py [--lines | --words | --chars] FILE [FILE ...]", file=sys.stderr)
        return 2
    mode = options[0] if options else "--lines"
    total, missing = 0, False
    for name in files:
        try:
            with open(name, encoding="utf-8", newline="") as f:
                text = f.read()
        except OSError:
            print(f"wc_tool: {name}: no such file", file=sys.stderr)
            missing = True
            continue
        count = text.count("\n") if mode == "--lines" else len(text.split()) if mode == "--words" else len(text)
        print(f"{count} {name}")
        total += count
    if len(files) > 1:
        print(f"{total} total")
    return 1 if missing else 0


sys.exit(main(sys.argv[1:]))
'''},
)

# ----------------------------------------------------------------------------- geometry
_add(
    "geometry", "contract",
    "Write the module geometry.py in /work with the exact interface described in SPEC.md. "
    "Running `python check.py` does a basic check.",
    {
        "SPEC.md": r'''# Geometry helpers

Write the module `geometry.py`. Other code already calls it, so the interface
below has to be matched exactly.

    area(shape, *, units="m")                  -> float
    perimeter(shape)                           -> float
    scale(shape, factor, *, origin=(0.0, 0.0)) -> dict
    bbox(shapes)                               -> BBox
    BBox                                       (a namedtuple)

A shape is a dict. A rectangle is `{"type": "rect", "x": .., "y": .., "w": ..,
"h": ..}` where (x, y) is its lower-left corner. A circle is `{"type": "circle",
"cx": .., "cy": .., "r": ..}`.

## Rules
1. `units` and `origin` are keyword-only. Passing them by position, as in
   `area(shape, "cm")` or `scale(shape, 2, (1, 1))`, must raise `TypeError`.
2. `area` returns square metres for `units="m"` and square centimetres for
   `units="cm"` (10 000 times as much). Any other unit raises `ValueError`.
3. `area` and `perimeter` always return a `float`, also when the inputs are ints.
4. A shape whose `"type"` is not `"rect"` or `"circle"` raises `ValueError` in
   every function.
5. `scale` returns a new shape dict of the same type and leaves its argument
   unchanged. Sizes (`w`, `h`, `r`) are multiplied by `factor`. Positions move
   away from or toward `origin`: a coordinate p becomes
   `origin + (p - origin) * factor`, separately for x and y.
6. `factor` must be a number greater than 0, otherwise `ValueError`.
7. `BBox` is a `collections.namedtuple` named `BBox` with the fields
   `min_x, min_y, max_x, max_y`, in that order, and it is importable as
   `geometry.BBox`.
8. `bbox(shapes)` returns the smallest `BBox` that contains all shapes in the
   list. An empty list raises `ValueError`.
''' + _GRADING_CONTRACT,
        "check.py": r'''import sys
try:
    import geometry
    ok = abs(geometry.area({"type": "rect", "x": 0, "y": 0, "w": 2, "h": 3}) - 6.0) < 1e-9
except Exception as exc:
    print("CHECK FAILED:", repr(exc))
    sys.exit(1)
if not ok:
    print("CHECK FAILED: wrong area for a 2 x 3 rectangle")
    sys.exit(1)
print("ALL CHECKS PASSED")
''',
    },
    r'''
import math
import geometry as g
rect = {"type": "rect", "x": 1, "y": 2, "w": 4, "h": 2}
circ = {"type": "circle", "cx": -1, "cy": 0, "r": 2}
a = g.area(rect)
check("rect area is a float", a == 8.0 and isinstance(a, float))
check("circle area", abs(g.area(circ) - 4 * math.pi) < 1e-9)
check("units cm", g.area(rect, units="cm") == 80000.0)
check("unknown unit rejected", raises(ValueError, g.area, rect, units="km"))
check("units is keyword-only", raises(TypeError, g.area, rect, "cm"))
p = g.perimeter(rect)
check("perimeters are floats", p == 12.0 and isinstance(p, float) and abs(g.perimeter(circ) - 4 * math.pi) < 1e-9)
for fn in (g.area, g.perimeter, lambda s: g.scale(s, 2), lambda s: g.bbox([s])):
    check("unknown shape type rejected", raises(ValueError, fn, {"type": "tri", "x": 0}))
s = g.scale(rect, 2)
check("scale about default origin", s == {"type": "rect", "x": 2, "y": 4, "w": 8, "h": 4})
check("scale leaves the argument unchanged and returns a new dict", rect == {"type": "rect", "x": 1, "y": 2, "w": 4, "h": 2} and s is not rect)
check("scale about a given origin", g.scale(rect, 3, origin=(1, 2)) == {"type": "rect", "x": 1, "y": 2, "w": 12, "h": 6})
check("scale circle", g.scale(circ, 0.5, origin=(1, 0)) == {"type": "circle", "cx": 0, "cy": 0, "r": 1})
check("origin is keyword-only", raises(TypeError, g.scale, rect, 2, (1, 1)))
for bad in (0, -1):
    check(f"factor {bad} rejected", raises(ValueError, g.scale, rect, bad))
check("BBox is a namedtuple with the documented fields", getattr(g.BBox, "_fields", None) == ("min_x", "min_y", "max_x", "max_y") and g.BBox.__name__ == "BBox")
b = g.bbox([rect, circ])
check("bbox value and type", isinstance(b, g.BBox) and tuple(b) == (-3, -2, 5, 4))
check("bbox of empty list rejected", raises(ValueError, g.bbox, []))
finish()
''',
    {"geometry.py": r'''import math
from collections import namedtuple

BBox = namedtuple("BBox", ["min_x", "min_y", "max_x", "max_y"])


def _kind(shape):
    kind = shape.get("type") if isinstance(shape, dict) else None
    if kind not in ("rect", "circle"):
        raise ValueError(f"unknown shape type: {kind!r}")
    return kind


def area(shape, *, units="m"):
    kind = _kind(shape)
    if units not in ("m", "cm"):
        raise ValueError(f"unknown units: {units!r}")
    value = shape["w"] * shape["h"] if kind == "rect" else math.pi * shape["r"] ** 2
    return float(value * (10000 if units == "cm" else 1))


def perimeter(shape):
    if _kind(shape) == "rect":
        return float(2 * (shape["w"] + shape["h"]))
    return float(2 * math.pi * shape["r"])


def scale(shape, factor, *, origin=(0.0, 0.0)):
    kind = _kind(shape)
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 0:
        raise ValueError("factor must be a number greater than 0")
    ox, oy = origin
    if kind == "rect":
        return {"type": "rect", "x": ox + (shape["x"] - ox) * factor, "y": oy + (shape["y"] - oy) * factor,
                "w": shape["w"] * factor, "h": shape["h"] * factor}
    return {"type": "circle", "cx": ox + (shape["cx"] - ox) * factor,
            "cy": oy + (shape["cy"] - oy) * factor, "r": shape["r"] * factor}


def _extent(shape):
    if _kind(shape) == "rect":
        return shape["x"], shape["y"], shape["x"] + shape["w"], shape["y"] + shape["h"]
    return (shape["cx"] - shape["r"], shape["cy"] - shape["r"],
            shape["cx"] + shape["r"], shape["cy"] + shape["r"])


def bbox(shapes):
    extents = [_extent(s) for s in shapes]
    if not extents:
        raise ValueError("no shapes")
    return BBox(min(e[0] for e in extents), min(e[1] for e in extents),
                max(e[2] for e in extents), max(e[3] for e in extents))
'''},
)

# ----------------------------------------------------------------------------- export
_add(
    "export", "contract",
    "Write export.py in /work as described in SPEC.md, so that `python export.py people.json people.csv` "
    "writes the CSV file. Running `python check.py` checks it on people.json.",
    {
        "SPEC.md": r'''# CSV export

Write `export.py`. It is run as

    python export.py INPUT.json OUTPUT.csv

INPUT.json is a JSON array of person objects (`people.json` is a sample). The
CSV file is imported by another system that is strict about the format.

## Input fields
`id` (integer), `first_name`, `last_name`, `email`, `birth_date` (written
`DD/MM/YYYY`), `active` (true or false). `email`, `birth_date` and `active` may
be missing or `null`.

## Rules
1. The first line is the header, exactly
   `id,last_name,first_name,email,birth_date,active` (note the column order).
2. Every line, including the header and the last line, ends with `\r\n`
   (carriage return + line feed). The file is UTF-8 without a byte order mark.
3. Rows are sorted by last name, then first name, ignoring letter case, and
   finally by id.
4. `first_name` and `last_name` are trimmed of surrounding spaces. Their letter
   case is kept as it is.
5. `email` is trimmed and written in lower case. A missing or `null` email is an
   empty field.
6. `birth_date` is written as an ISO date, `YYYY-MM-DD`. A missing or `null`
   date is an empty field.
7. `active` is written as `yes` or `no`. Missing or `null` means `no`.
8. A field is wrapped in double quotes only when it contains a comma, a double
   quote or a line break. A double quote inside a quoted field is written twice.
   No other field is quoted.
''' + _GRADING_CONTRACT,
        "people.json": json.dumps([
            {"id": 1, "first_name": "Maria", "last_name": "Lopez", "email": "Maria@Example.com",
             "birth_date": "07/03/1988", "active": True},
            {"id": 2, "first_name": "Ken", "last_name": "Abe", "email": None, "birth_date": "22/11/1975",
             "active": False}], indent=1),
        "check.py": r'''import os, subprocess, sys, tempfile
out = os.path.join(tempfile.mkdtemp(), "people.csv")
r = subprocess.run([sys.executable, "export.py", "people.json", out], capture_output=True, text=True)
if r.returncode != 0 or not os.path.exists(out):
    print("CHECK FAILED: export.py did not produce the output file\n" + r.stderr[-300:])
    sys.exit(1)
lines = open(out, encoding="utf-8").read().split("\n")
if not lines[0].startswith("id,") or len([l for l in lines if l.strip()]) != 3:
    print("CHECK FAILED: expected a header and two rows")
    sys.exit(1)
print("ALL CHECKS PASSED")
''',
    },
    r'''
import pathlib, shutil
tmp = pathlib.Path("/tmp/verify_export"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
people = [
    {"id": 3, "first_name": "Ann", "last_name": "o'Neil", "email": " Ann@X.org ", "birth_date": "05/11/1990", "active": True},
    {"id": 1, "first_name": " bob ", "last_name": "Smith, Jr.", "email": None, "active": False},
    {"id": 2, "first_name": "Zoë", "last_name": "adams", "birth_date": "31/01/2001", "active": True},
    {"id": 5, "first_name": 'Al "Ace"', "last_name": "Smith", "email": "AL@x.org", "birth_date": None},
    {"id": 4, "first_name": "ann", "last_name": "O'Neil", "email": "a2@x.org", "birth_date": "01/02/1985", "active": True},
]
(tmp / "in.json").write_text(json.dumps(people), encoding="utf-8")
r = sh(f"python export.py {tmp}/in.json {tmp}/out.csv")
check("exit code 0", r.returncode == 0)
raw = (tmp / "out.csv").read_bytes() if (tmp / "out.csv").exists() else b""
rows = [
    "id,last_name,first_name,email,birth_date,active",
    "2,adams,Zoë,,2001-01-31,yes",
    "3,o'Neil,Ann,ann@x.org,1990-11-05,yes",
    "4,O'Neil,ann,a2@x.org,1985-02-01,yes",
    '5,Smith,"Al ""Ace""",al@x.org,,no',
    '1,"Smith, Jr.",bob,,,no',
]
text = raw.decode("utf-8", "replace")
got = text.replace("\r\n", "\n").split("\n")
check("no byte order mark", not raw.startswith(b"\xef\xbb\xbf"))
check("header exact and in the documented column order", got[:1] == rows[:1])
check("CRLF line endings on every line including the last", text.endswith("\r\n") and text.count("\r\n") == text.count("\n") == len(rows))
check("row order: last name, first name (case-insensitive), then id", [l.split(",")[0] for l in got[1:6]] == ["2", "3", "4", "5", "1"])
check("emails lower-cased and trimmed, missing email empty", "ann@x.org" in text and "al@x.org" in text and "None" not in text and "null" not in text)
check("ISO dates and empty missing dates, active yes/no", got[1:2] == rows[1:2])
check("quoting only when needed, quotes doubled", got[4:6] == rows[4:6])
check("whole file exact", raw == ("\r\n".join(rows) + "\r\n").encode("utf-8"))
finish()
''',
    {"export.py": r'''import csv
import json
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        people = json.load(f)
    rows = []
    for p in people:
        last = (p.get("last_name") or "").strip()
        first = (p.get("first_name") or "").strip()
        email = (p.get("email") or "").strip().lower()
        born = p.get("birth_date") or ""
        if born:
            day, month, year = born.strip().split("/")
            born = f"{year}-{month}-{day}"
        rows.append(((last.lower(), first.lower(), p["id"]),
                     [p["id"], last, first, email, born, "yes" if p.get("active") else "no"]))
    rows.sort(key=lambda r: r[0])
    with open(dst, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(["id", "last_name", "first_name", "email", "birth_date", "active"])
        for _, row in rows:
            writer.writerow(row)


main()
'''},
)

# ----------------------------------------------------------------------------- config
_add(
    "config", "contract",
    "Write migrate_config.py in /work as described in SPEC.md, so that `python migrate_config.py old.ini new.json` "
    "converts the configuration. Running `python check.py` checks it on old.ini.",
    {
        "SPEC.md": r'''# Configuration migration

Write `migrate_config.py`. It is run as

    python migrate_config.py OLD.ini NEW.json

and converts an old INI configuration (`old.ini` is a sample) into the new JSON
format.

## Rules
1. NEW.json holds a JSON object with exactly four keys: `"server"`,
   `"database"`, `"features"` and `"extra"`. It is written with 2-space
   indentation, with the keys of every object sorted alphabetically, and it ends
   with exactly one newline.
2. Key names in the INI file are case-insensitive (`Hostname` is `hostname`).
3. `"server"` is `{"host": .., "port": .., "tls": ..}`. `host` comes from
   `hostname` in `[server]` (default `"localhost"`). `tls` is a boolean from
   `use_ssl` (default false). `port` is the integer in `port`; when `port` is
   absent it is 443 if `tls` is true and 80 otherwise.
4. Booleans are written `yes/true/1/on` or `no/false/0/off` in any letter case.
5. `"database"` is `{"url": .., "pool": ..}`. `url` is
   `postgres://<db_host>:<db_port>/<db_name>` with the defaults `localhost`,
   `5432` and `app`. `pool` is the integer in `pool_size` (default 5).
6. `"features"` is the alphabetically sorted list of the key names in
   `[features]` whose value is true.
7. Two deprecated keys are dropped without a trace: `legacy_mode` in `[server]`
   and `db_driver` in `[database]`.
8. Every other key is preserved in `"extra"` as `"<section>.<key>": "<value>"`,
   with the value kept as a string. This covers unknown keys in `[server]` and
   `[database]` and all keys of any other section. `"extra"` is `{}` when there
   is nothing to preserve.
9. A value that is not a valid boolean where one is expected, or not an integer
   where one is expected, is an error: print a message to stderr, exit with
   code 3 and do not write NEW.json.
''' + _GRADING_CONTRACT,
        "old.ini": "[server]\nhostname = intranet.local\nport = 8080\nuse_ssl = no\n\n[database]\n"
                   "db_host = db1\ndb_name = main\n\n[features]\nreports = yes\n",
        "check.py": r'''import json, os, subprocess, sys, tempfile
out = os.path.join(tempfile.mkdtemp(), "new.json")
r = subprocess.run([sys.executable, "migrate_config.py", "old.ini", out], capture_output=True, text=True)
if r.returncode != 0 or not os.path.exists(out):
    print("CHECK FAILED: migrate_config.py did not produce the output file\n" + r.stderr[-300:])
    sys.exit(1)
data = json.load(open(out))
if data.get("server", {}).get("host") != "intranet.local" or data.get("server", {}).get("port") != 8080:
    print("CHECK FAILED: unexpected server section")
    sys.exit(1)
print("ALL CHECKS PASSED")
''',
    },
    r'''
import pathlib, shutil
tmp = pathlib.Path("/tmp/verify_config"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
(tmp / "a.ini").write_text("[server]\nHostname = example.org\nuse_ssl = ON\nlegacy_mode = yes\ntimeout = 30\n\n"
                           "[database]\ndb_name = shop\npool_size = 12\ndb_driver = psycopg\n\n"
                           "[features]\nbeta = yes\ndark_mode = 0\nSearch = True\n\n[logging]\nlevel = debug\n")
want = {"server": {"host": "example.org", "port": 443, "tls": True},
        "database": {"url": "postgres://localhost:5432/shop", "pool": 12},
        "features": ["beta", "search"],
        "extra": {"server.timeout": "30", "logging.level": "debug"}}
r = sh(f"python migrate_config.py {tmp}/a.ini {tmp}/a.json")
check("exit code 0", r.returncode == 0)
text = (tmp / "a.json").read_text() if (tmp / "a.json").exists() else "{}"
try:
    got = json.loads(text)
except Exception:
    got = {}
check("server: case-insensitive key, boolean spelling, port default follows tls", got.get("server") == want["server"])
check("database: url defaults and integer pool", got.get("database") == want["database"])
check("features: only true ones, sorted, lower-case", got.get("features") == want["features"])
check("extra: unknown keys kept as strings, deprecated keys dropped", got.get("extra") == want["extra"])
check("exactly four top-level keys", sorted(got) == ["database", "extra", "features", "server"])
check("format: 2-space indent, sorted keys, one trailing newline", text == json.dumps(want, indent=2, sort_keys=True) + "\n")
(tmp / "b.ini").write_text("[server]\nport = 9000\n")
r = sh(f"python migrate_config.py {tmp}/b.ini {tmp}/b.json")
try:
    got = json.loads((tmp / "b.json").read_text())
except Exception:
    got = {}
check("all defaults, empty extra", got == {"server": {"host": "localhost", "port": 9000, "tls": False},
      "database": {"url": "postgres://localhost:5432/app", "pool": 5}, "features": [], "extra": {}})
for name, body in (("c", "[server]\nuse_ssl = maybe\n"), ("d", "[database]\npool_size = many\n"), ("e", "[features]\nx = perhaps\n")):
    (tmp / f"{name}.ini").write_text(body)
    r = sh(f"python migrate_config.py {tmp}/{name}.ini {tmp}/{name}.json")
    check(f"invalid value in {body.split()[0]}: exit 3, stderr message, no output file",
          r.returncode == 3 and r.stderr.strip() != "" and not (tmp / f"{name}.json").exists())
finish()
''',
    {"migrate_config.py": r'''import configparser
import json
import sys

TRUE, FALSE = {"yes", "true", "1", "on"}, {"no", "false", "0", "off"}


def boolean(value):
    value = value.strip().lower()
    if value in TRUE:
        return True
    if value in FALSE:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(src)
    sections = {name: dict(parser[name]) for name in parser.sections()}
    server = sections.pop("server", {})
    database = sections.pop("database", {})
    features = sections.pop("features", {})
    try:
        tls = boolean(server.pop("use_ssl", "no"))
        port = int(server.pop("port")) if "port" in server else (443 if tls else 80)
        out_server = {"host": server.pop("hostname", "localhost"), "port": port, "tls": tls}
        url = "postgres://{}:{}/{}".format(database.pop("db_host", "localhost"),
                                           int(database.pop("db_port", "5432")),
                                           database.pop("db_name", "app"))
        out_database = {"url": url, "pool": int(database.pop("pool_size", "5"))}
        enabled = sorted(name for name, value in features.items() if boolean(value))
    except ValueError as exc:
        print(f"migrate_config: {exc}", file=sys.stderr)
        return 3
    server.pop("legacy_mode", None)
    database.pop("db_driver", None)
    extra = {}
    for section, rest in [("server", server), ("database", database)] + list(sections.items()):
        for key, value in rest.items():
            extra[f"{section}.{key}"] = value
    with open(dst, "w") as f:
        json.dump({"server": out_server, "database": out_database, "features": enabled, "extra": extra},
                  f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


sys.exit(main())
'''},
)

# ----------------------------------------------------------------------------- migrate (idempotent)
_add(
    "migrate", "contract",
    "Write migrate.py in /work as described in SPEC.md. It upgrades a data file in place: "
    "`python migrate.py db.json`. Running `python check.py` tries it on a copy of db.json.",
    {
        "SPEC.md": r'''# In-place data migration

Write `migrate.py`. It is run as

    python migrate.py PATH

and upgrades the JSON file at PATH from schema version 1 to schema version 2,
in place. `db.json` is a sample version-1 file. The script runs unattended on
many machines and may be run more than once on the same file.

## Version 1
    {"schema_version": 1,
     "users": [{"id": 7, "name": "Ada Lovelace", "mail": "ada@x.org", "tags": "math, code"}]}

## Version 2
    {"schema_version": 2,
     "users": [{"id": 7, "first_name": "Ada", "last_name": "Lovelace",
                "email": "ada@x.org", "tags": ["code", "math"]}]}

## Rules
1. `first_name` is the part of `name` before the first space. `last_name` is
   everything after that space, and is `""` when the name has no space.
2. `email` is the old `mail` value.
3. `tags` becomes a list: split the old string at commas, trim each piece, drop
   empty pieces, remove duplicates and sort.
4. Users are sorted by `id`. The keys of a user object appear in this order:
   `id, first_name, last_name, email, tags`. The top-level keys appear in the
   order `schema_version, users`.
5. The file is written as UTF-8 with 2-space indentation. Non-ASCII characters
   are written as themselves, not as `\u` escapes. The file ends with exactly
   one newline.
6. Before a version-1 file is changed, an exact byte-for-byte copy of it is saved
   as PATH + `.bak` (for `db.json` that is `db.json.bak`).
7. The script is idempotent. When the file is already version 2, nothing is
   changed: the file and any existing `.bak` file stay byte-for-byte the same.
   The script prints `already migrated` and exits with code 0.
8. For any other `schema_version`, or when it is missing, the script prints a
   message to stderr and exits with code 4. It changes nothing and creates no
   `.bak` file.
''' + _GRADING_CONTRACT,
        "db.json": json.dumps({"schema_version": 1, "users": [
            {"id": 2, "name": "Grace Hopper", "mail": "grace@x.org", "tags": "navy, code"},
            {"id": 1, "name": "Ada Lovelace", "mail": "ada@x.org", "tags": "math, code"}]}, indent=1),
        "check.py": r'''import json, os, shutil, subprocess, sys, tempfile
path = os.path.join(tempfile.mkdtemp(), "db.json")
shutil.copy("db.json", path)
r = subprocess.run([sys.executable, "migrate.py", path], capture_output=True, text=True)
if r.returncode != 0:
    print("CHECK FAILED: migrate.py exited with an error\n" + r.stderr[-300:])
    sys.exit(1)
data = json.load(open(path))
if data.get("schema_version") != 2 or "first_name" not in data["users"][0]:
    print("CHECK FAILED: the copy was not upgraded to version 2")
    sys.exit(1)
print("ALL CHECKS PASSED")
''',
    },
    r'''
import pathlib, shutil
tmp = pathlib.Path("/tmp/verify_migrate"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
v1 = ('{"schema_version": 1,   "users": [\n  {"id": 9, "name": "Cher", "mail": "c@x.org", "tags": ""},\n'
      '  {"id": 3, "name": "José Ángel Pérez", "mail": "jose@x.org", "tags": " b, a,,b "}\n]}\n')
want = {"schema_version": 2, "users": [
    {"id": 3, "first_name": "José", "last_name": "Ángel Pérez", "email": "jose@x.org", "tags": ["a", "b"]},
    {"id": 9, "first_name": "Cher", "last_name": "", "email": "c@x.org", "tags": []}]}
want_text = json.dumps(want, indent=2, ensure_ascii=False) + "\n"
db = tmp / "data.json"
db.write_bytes(v1.encode("utf-8"))
r = sh(f"python migrate.py {db}")
check("first run exits 0", r.returncode == 0)
text = db.read_bytes().decode("utf-8", "replace")
try:
    got = json.loads(text)
except Exception:
    got = {}
check("content: name split at the first space, mail renamed, tags cleaned, users sorted by id", got == want)
check("format: key order, 2-space indent, real non-ASCII characters, one trailing newline", text == want_text)
bak = pathlib.Path(str(db) + ".bak")
check("backup is an exact byte copy of the original", bak.exists() and bak.read_bytes() == v1.encode("utf-8"))
r = sh(f"python migrate.py {db}")
check("second run: exit 0 and prints 'already migrated'", r.returncode == 0 and "already migrated" in r.stdout)
check("second run changes nothing", db.read_bytes().decode("utf-8", "replace") == text
      and bak.exists() and bak.read_bytes() == v1.encode("utf-8"))
for name, body in (("v3", '{"schema_version": 3, "users": []}'), ("none", '{"users": []}')):
    other = tmp / f"{name}.json"
    other.write_text(body)
    r = sh(f"python migrate.py {other}")
    check(f"unsupported version ({name}): exit 4, stderr message, nothing changed, no backup",
          r.returncode == 4 and r.stderr.strip() != "" and other.read_text() == body
          and not pathlib.Path(str(other) + ".bak").exists())
finish()
''',
    {"migrate.py": r'''import json
import sys


def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        raw = f.read()
    data = json.loads(raw.decode("utf-8"))
    version = data.get("schema_version")
    if version == 2:
        print("already migrated")
        return 0
    if version != 1:
        print(f"migrate: unsupported schema_version {version!r}", file=sys.stderr)
        return 4
    users = []
    for user in sorted(data.get("users", []), key=lambda u: u["id"]):
        first, _, last = user.get("name", "").partition(" ")
        tags = sorted({t.strip() for t in (user.get("tags") or "").split(",") if t.strip()})
        users.append({"id": user["id"], "first_name": first, "last_name": last,
                      "email": user.get("mail"), "tags": tags})
    with open(path + ".bak", "wb") as f:
        f.write(raw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 2, "users": users}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return 0


sys.exit(main())
'''},
)
