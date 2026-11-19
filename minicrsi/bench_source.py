"""Source benchmark: 12 "make it correct" coding tasks.

Each task gives the agent a one-line instruction; the real requirements are in
SPEC.md inside the working directory. A hidden verifier checks every rule.
Reference solutions exist only so `python -m minicrsi.selfcheck` can prove that
each task is solvable and that its verifier rejects the untouched starting state.
"""
from .tasks import Task

TASKS: list[Task] = []


def _add(name, kind, instruction, files, verify, solution):
    TASKS.append(Task(name=name, kind=kind, instruction=instruction, files=files,
                      verify=verify, solution=solution))


# ----------------------------------------------------------------------------- roman
_add(
    "roman", "library",
    "Implement roman.py in /work as described in SPEC.md. A few public tests are in "
    "test_public.py (run them with: python -m unittest -q test_public).",
    {
        "SPEC.md": r'''# Roman numerals

Create `roman.py` in /work with two functions.

## to_roman(n) -> str
Convert an integer to a Roman numeral in the standard subtractive form
(4 = IV, 9 = IX, 40 = XL, 90 = XC, 400 = CD, 900 = CM).

## from_roman(s) -> int
Convert a Roman numeral back to an integer.

## Rules
1. `to_roman` accepts integers from 1 to 3999. Any other integer raises `ValueError`.
2. `to_roman` raises `TypeError` for anything that is not an `int`. `bool` is not
   accepted either: `to_roman(True)` raises `TypeError`. So does `to_roman(3.0)`.
3. `from_roman` accepts only the exact string that `to_roman` would produce.
   Everything else raises `ValueError`. These must all be rejected:
   `"IIII"`, `"VX"`, `"IC"`, `"MMMM"`, `"iv"` (lower case), `" IV"` (whitespace), `""`.
4. `from_roman` raises `TypeError` when its argument is not a `str`.
''',
        "test_public.py": r'''import unittest
import roman


class PublicTests(unittest.TestCase):
    def test_to_roman(self):
        self.assertEqual(roman.to_roman(4), "IV")
        self.assertEqual(roman.to_roman(1994), "MCMXCIV")

    def test_from_roman(self):
        self.assertEqual(roman.from_roman("MCMXCIV"), 1994)

    def test_range(self):
        with self.assertRaises(ValueError):
            roman.to_roman(0)
''',
    },
    r'''
import roman
ok = True
for n in range(1, 4000):
    if roman.from_roman(roman.to_roman(n)) != n:
        ok = False
        break
check("round trip for 1..3999", ok)
check("known values", roman.to_roman(1994) == "MCMXCIV" and roman.to_roman(3999) == "MMMCMXCIX"
      and roman.to_roman(444) == "CDXLIV")
for bad in (0, -1, 4000):
    check(f"to_roman({bad}) raises ValueError", raises(ValueError, roman.to_roman, bad))
for bad in (True, 3.0, "3", None):
    check(f"to_roman({bad!r}) raises TypeError", raises(TypeError, roman.to_roman, bad))
for bad in ("IIII", "VX", "IC", "MMMM", "iv", " IV", "", "IVI", "XM", "A"):
    check(f"from_roman({bad!r}) raises ValueError", raises(ValueError, roman.from_roman, bad))
for bad in (4, None, b"IV"):
    check(f"from_roman({bad!r}) raises TypeError", raises(TypeError, roman.from_roman, bad))
finish()
''',
    {"roman.py": r'''_PAIRS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
          (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
_VAL = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def to_roman(n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("int required")
    if not 1 <= n <= 3999:
        raise ValueError("out of range")
    out = []
    for value, sym in _PAIRS:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def from_roman(s):
    if not isinstance(s, str):
        raise TypeError("str required")
    if not s or any(c not in _VAL for c in s):
        raise ValueError("invalid numeral")
    total = 0
    for i, c in enumerate(s):
        v = _VAL[c]
        total += -v if i + 1 < len(s) and _VAL[s[i + 1]] > v else v
    if not 1 <= total <= 3999 or to_roman(total) != s:
        raise ValueError("invalid numeral")
    return total
'''},
)

# ----------------------------------------------------------------------------- intervals
_add(
    "intervals", "library",
    "Implement intervals.py in /work as described in SPEC.md. Public tests: "
    "python -m unittest -q test_public",
    {
        "SPEC.md": r'''# Integer intervals

Create `intervals.py` with three functions. An interval is a pair `(a, b)` of
integers and stands for every integer from a to b, both ends included.

    merge(intervals)          -> list of tuples
    total_covered(intervals)  -> int
    gaps(intervals)           -> list of tuples

## Rules
1. `merge` returns the shortest list of intervals that covers exactly the same
   integers, sorted by start. Every element of the result is a tuple.
2. Overlapping intervals are merged. Adjacent intervals are merged too: `(1, 2)`
   and `(3, 4)` together cover 1, 2, 3, 4 with nothing missing, so the result is
   `(1, 4)`. `(1, 2)` and `(4, 5)` stay separate because 3 is missing.
3. An interval with a > b is invalid: raise `ValueError`. This applies to all
   three functions.
4. The input list must not be modified. Its items may be lists or tuples.
5. `total_covered` returns how many distinct integers are covered:
   `total_covered([(1, 3), (2, 5)])` is 5.
6. `gaps` returns the runs of integers that lie between the covered parts, as
   intervals sorted by start: `gaps([(1, 2), (6, 8), (12, 12)])` is
   `[(3, 5), (9, 11)]`. If the merged result has fewer than two intervals there
   are no gaps.
7. Empty input gives `[]`, `0` and `[]`.
''',
        "test_public.py": r'''import unittest
import intervals


class PublicTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(intervals.merge([(1, 3), (2, 6), (8, 10)]), [(1, 6), (8, 10)])

    def test_total(self):
        self.assertEqual(intervals.total_covered([(1, 3), (2, 5)]), 5)
''',
    },
    r'''
import intervals as iv
check("overlap", iv.merge([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)])
check("adjacent merged", iv.merge([(3, 4), (1, 2)]) == [(1, 4)])
check("non adjacent kept", iv.merge([(4, 5), (1, 2)]) == [(1, 2), (4, 5)])
check("contained", iv.merge([(1, 10), (2, 3), (11, 11)]) == [(1, 11)])
res = iv.merge([[5, 7], [1, 1]])
check("tuples returned for list input", res == [(1, 1), (5, 7)] and all(isinstance(x, tuple) for x in res))
src = [(5, 9), (1, 2)]
iv.merge(src); iv.gaps(src); iv.total_covered(src)
check("input not modified", src == [(5, 9), (1, 2)])
for fn in (iv.merge, iv.total_covered, iv.gaps):
    check(f"{fn.__name__} rejects a > b", raises(ValueError, fn, [(1, 2), (5, 4)]))
check("total covered", iv.total_covered([(1, 3), (2, 5)]) == 5 and iv.total_covered([(-2, 0), (5, 5)]) == 4)
check("gaps", iv.gaps([(1, 2), (6, 8), (12, 12)]) == [(3, 5), (9, 11)])
check("no gap between adjacent", iv.gaps([(1, 2), (3, 4)]) == [])
check("gaps unsorted input", iv.gaps([(10, 12), (1, 1)]) == [(2, 9)])
check("empty", iv.merge([]) == [] and iv.total_covered([]) == 0 and iv.gaps([]) == [])
finish()
''',
    {"intervals.py": r'''def _sorted_valid(intervals):
    items = []
    for a, b in intervals:
        if a > b:
            raise ValueError("interval start is after its end")
        items.append((a, b))
    return sorted(items)


def merge(intervals):
    out = []
    for a, b in _sorted_valid(intervals):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def total_covered(intervals):
    return sum(b - a + 1 for a, b in merge(intervals))


def gaps(intervals):
    merged = merge(intervals)
    return [(merged[i][1] + 1, merged[i + 1][0] - 1) for i in range(len(merged) - 1)]
'''},
)

# ----------------------------------------------------------------------------- csv_stats
_add(
    "csv_stats", "script",
    "Write stats.py in /work so that `python stats.py INPUT.csv OUTPUT.json` produces the summary "
    "described in SPEC.md. A sample input is data.csv.",
    {
        "SPEC.md": r'''# Column statistics

Write `stats.py`. It is run as

    python stats.py INPUT.csv OUTPUT.json

and writes a JSON summary of the CSV file. `data.csv` is a sample input. The
grader runs your script on other CSV files that follow the same rules.

## Input
The first row is the header. Fields may be wrapped in double quotes and may then
contain commas. A field that is empty or contains only spaces is "blank".

## Output
A JSON object: `{"rows": <number of data rows>, "columns": {"<name>": {...}, ...}}`

## Rules
1. A column is numeric when it has at least one non-blank value and every
   non-blank value can be parsed by `float()`.
2. Numeric column:
   `{"type": "numeric", "count": n, "mean": m, "median": d, "min": lo, "max": hi}`.
   `count` is the number of non-blank values. `mean` and `median` are rounded to
   3 decimal places with `round()`.
3. Any other column with at least one non-blank value is a text column:
   `{"type": "text", "count": n, "distinct": k}`, where `distinct` counts the
   different non-blank values after stripping surrounding spaces.
4. A column whose values are all blank is `{"type": "empty", "count": 0}`.
5. Blank values are skipped in every calculation. `rows` counts every data row,
   including rows that contain blanks.
6. The median of an even number of values is the mean of the two middle values.
7. On success the exit code is 0. If INPUT.csv does not exist, print a message to
   stderr, exit with code 2, and do not create OUTPUT.json.
''',
        "data.csv": 'name,age,city,score,notes\n"Smith, Anna",34,Paris,88.5,\nBob,41,"New York, NY",91,ok\n'
                    'Cara,,Paris,79.25,\nDan,29, ,x,late\n',
    },
    r'''
import pathlib, shutil
tmp = pathlib.Path("/tmp/verify_stats")
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)
(tmp / "in.csv").write_text('id,label,amount,empty,qty\n1,"a, b",10.5,,3\n2,c,,," 4 "\n3,"a, b",-2,,\n4, c ,7.25,,10\n')
r = sh(f"python stats.py {tmp}/in.csv {tmp}/out.json")
check("exit code 0 on success", r.returncode == 0)
out = json.loads((tmp / "out.json").read_text()) if (tmp / "out.json").exists() else {}
cols = out.get("columns", {})
check("rows counted", out.get("rows") == 4)

def close(a, b):
    return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(a - b) < 1e-6

def numeric(name, count, mean, median, lo, hi):
    c = cols.get(name, {})
    return (c.get("type") == "numeric" and c.get("count") == count and close(c.get("mean"), mean)
            and close(c.get("median"), median) and close(c.get("min"), lo) and close(c.get("max"), hi))

check("numeric column id", numeric("id", 4, 2.5, 2.5, 1, 4))
check("numeric column with blanks", numeric("amount", 3, 5.25, 7.25, -2, 10.5))
check("numeric column with padded values and rounding", numeric("qty", 3, 5.667, 4, 3, 10))
check("text column with quoted commas", cols.get("label") == {"type": "text", "count": 4, "distinct": 2})
check("all-blank column", cols.get("empty") == {"type": "empty", "count": 0})
r = sh(f"python stats.py {tmp}/missing.csv {tmp}/none.json")
check("missing input exits with code 2", r.returncode == 2)
check("missing input writes to stderr", r.stderr.strip() != "")
check("missing input creates no output", not (tmp / "none.json").exists())
finish()
''',
    {"stats.py": r'''import csv
import json
import os
import statistics
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    if not os.path.isfile(src):
        print(f"stats: {src}: no such file", file=sys.stderr)
        sys.exit(2)
    with open(src, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    columns = {}
    for i, name in enumerate(header):
        values = [r[i].strip() for r in data if i < len(r) and r[i].strip() != ""]
        if not values:
            columns[name] = {"type": "empty", "count": 0}
            continue
        try:
            nums = [float(v) for v in values]
        except ValueError:
            columns[name] = {"type": "text", "count": len(values), "distinct": len(set(values))}
            continue
        columns[name] = {"type": "numeric", "count": len(nums),
                         "mean": round(sum(nums) / len(nums), 3),
                         "median": round(statistics.median(nums), 3),
                         "min": min(nums), "max": max(nums)}
    with open(dst, "w") as f:
        json.dump({"rows": len(data), "columns": columns}, f)


main()
'''},
)

# ----------------------------------------------------------------------------- lru
_add(
    "lru", "library",
    "Implement the LRUCache class in /work/lru.py as described in SPEC.md. Public tests: "
    "python -m unittest -q test_public",
    {
        "SPEC.md": r'''# LRU cache

Create `lru.py` with a class `LRUCache`.

    LRUCache(capacity)
    cache.get(key, default=None)
    cache.put(key, value)
    cache.peek(key, default=None)
    cache.keys()
    cache.resize(capacity)
    cache.stats()
    len(cache)

## Rules
1. `capacity` must be an `int` of at least 1, otherwise `ValueError`. This holds
   for the constructor and for `resize`.
2. `get` returns the stored value and makes the key the most recently used one.
   For a missing key it returns `default`.
3. `put` inserts or updates. Either way the key becomes the most recently used.
   When a new key is inserted into a full cache, the least recently used key is
   evicted first.
4. `peek` returns the value (or `default`) without changing the order and
   without touching the statistics.
5. `keys()` returns a list of keys ordered from most recently used to least
   recently used.
6. `resize(n)` changes the capacity. If the cache holds more than n items, the
   least recently used ones are evicted until it fits.
7. `stats()` returns `{"hits": h, "misses": m, "evictions": e}`. A `get` on a
   present key is a hit, a `get` on a missing key is a miss, and every evicted
   key counts as one eviction, whether it was evicted by `put` or by `resize`.
   Updating an existing key with `put` is not an eviction.
8. `None` is a valid value. A `get` on a key whose value is `None` is a hit.
''',
        "test_public.py": r'''import unittest
from lru import LRUCache


class PublicTests(unittest.TestCase):
    def test_basic_eviction(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")
        c.put("c", 3)
        self.assertEqual(c.get("b"), None)
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(len(c), 2)
''',
    },
    r'''
from lru import LRUCache
for bad in (0, -1, 1.5, "2"):
    check(f"capacity {bad!r} rejected", raises(ValueError, LRUCache, bad))
c = LRUCache(2)
c.put("a", 1); c.put("b", 2)
check("get refreshes", c.get("a") == 1 and c.keys() == ["a", "b"])
c.put("c", 3)
check("lru evicted", c.peek("b") is None and c.keys() == ["c", "a"] and len(c) == 2)
check("peek keeps order", c.peek("a") == 1 and c.keys() == ["c", "a"])
check("peek default", c.peek("zz", 7) == 7)
c.put("a", 10)
check("update refreshes without eviction", c.keys() == ["a", "c"] and c.get("a") == 10)
check("miss default", c.get("nope", "d") == "d")
st = c.stats()
check("stats after puts", st == {"hits": 2, "misses": 1, "evictions": 1})
d = LRUCache(3)
for k in "xyz":
    d.put(k, k.upper())
d.get("x")
d.resize(1)
check("resize evicts lru", d.keys() == ["x"] and d.stats()["evictions"] == 2)
check("resize validates", raises(ValueError, d.resize, 0))
d.resize(2); d.put("n", None)
check("None is a value", d.get("n", "default") is None and d.stats()["hits"] == 2)
d.put("q", 1)
check("capacity after resize up", d.keys() == ["q", "n"])
finish()
''',
    {"lru.py": r'''from collections import OrderedDict

_MISSING = object()


def _check(capacity):
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("capacity must be an int >= 1")


class LRUCache:
    def __init__(self, capacity):
        _check(capacity)
        self._cap = capacity
        self._data = OrderedDict()   # oldest first
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def __len__(self):
        return len(self._data)

    def get(self, key, default=None):
        if key in self._data:
            self._stats["hits"] += 1
            self._data.move_to_end(key)
            return self._data[key]
        self._stats["misses"] += 1
        return default

    def peek(self, key, default=None):
        return self._data.get(key, default)

    def _shrink(self):
        while len(self._data) > self._cap:
            self._data.popitem(last=False)
            self._stats["evictions"] += 1

    def put(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        self._shrink()

    def keys(self):
        return list(reversed(self._data))

    def resize(self, capacity):
        _check(capacity)
        self._cap = capacity
        self._shrink()

    def stats(self):
        return dict(self._stats)
'''},
)

# ----------------------------------------------------------------------------- ledger (bug fix)
_LEDGER_BUGGY = r'''class Ledger:
    def __init__(self):
        self.accounts = {}
        self.log = {}

    def open(self, name, cents=0):
        self.accounts[name] = cents
        self.log[name] = []

    def balance(self, name):
        return self.accounts[name]

    def deposit(self, name, cents):
        if cents < 0:
            raise ValueError("bad amount")
        self.accounts[name] += cents
        self.log[name].append(("deposit", cents))

    def withdraw(self, name, cents):
        if cents < 0:
            raise ValueError("bad amount")
        if self.accounts[name] - cents < 0:
            raise ValueError("insufficient funds")
        self.accounts[name] -= cents
        self.log[name].append(("withdraw", cents))

    def transfer(self, src, dst, cents):
        self.withdraw(src, cents)
        self.deposit(dst, cents)

    def history(self, name):
        return self.log[name]

    def apply_interest(self, rate_bps):
        for name in self.accounts:
            self.accounts[name] += round(self.accounts[name] * rate_bps / 10000)
'''

_add(
    "ledger", "bugfix",
    "ledger.py in /work has several bugs. Fix it so that it follows every rule in SPEC.md. "
    "Public tests: python -m unittest -q test_public",
    {
        "ledger.py": _LEDGER_BUGGY,
        "SPEC.md": r'''# Ledger bug fix

`ledger.py` keeps account balances in whole cents. It has several bugs. Fix it so
that it follows every rule below. Keep the class name `Ledger` and the method
names and parameters as they are.

## Rules
1. Amounts are whole cents. `deposit`, `withdraw` and `transfer` accept only
   `int` amounts greater than zero. Zero, negative numbers, floats and bools
   raise `ValueError`.
2. `open(name, cents=0)` creates an account. Opening a name that already exists
   raises `ValueError`. The opening amount must be an `int` >= 0 (not a bool),
   otherwise `ValueError`.
3. A method that is given an account name that does not exist raises `KeyError`.
4. `withdraw` raises `ValueError` if the balance would go below zero.
5. `transfer(src, dst, cents)` is all-or-nothing: if it fails for any reason, no
   balance and no history changes at all. Transferring from an account to itself
   raises `ValueError`.
6. `history(name)` returns a list of `(kind, cents)` tuples in the order they
   happened. The kinds are `"deposit"`, `"withdraw"`, `"transfer_out"`,
   `"transfer_in"` and `"interest"`. A transfer adds `("transfer_out", cents)` to
   the source and `("transfer_in", cents)` to the destination. The caller may
   change the returned list without affecting the ledger.
7. `apply_interest(rate_bps)` adds `balance * rate_bps // 10000` cents to every
   account (always rounding down). An account that receives 0 cents gets no
   history entry; the others get `("interest", amount)`.
''',
        "test_public.py": r'''import unittest
from ledger import Ledger


class PublicTests(unittest.TestCase):
    def test_deposit_withdraw(self):
        led = Ledger()
        led.open("a", 100)
        led.deposit("a", 50)
        led.withdraw("a", 30)
        self.assertEqual(led.balance("a"), 120)

    def test_overdraft(self):
        led = Ledger()
        led.open("a", 10)
        with self.assertRaises(ValueError):
            led.withdraw("a", 11)
''',
    },
    r'''
from ledger import Ledger
led = Ledger()
led.open("a", 100); led.open("b")
for bad in (0, -5, 1.5, True):
    check(f"deposit {bad!r} rejected", raises(ValueError, led.deposit, "a", bad))
    check(f"withdraw {bad!r} rejected", raises(ValueError, led.withdraw, "a", bad))
    check(f"transfer {bad!r} rejected", raises(ValueError, led.transfer, "a", "b", bad))
check("reopen rejected", raises(ValueError, led.open, "a"))
check("negative opening rejected", raises(ValueError, Ledger().open, "x", -1))
check("bool opening rejected", raises(ValueError, Ledger().open, "x", True))
check("unknown account KeyError", raises(KeyError, led.balance, "zz") and raises(KeyError, led.deposit, "zz", 5)
      and raises(KeyError, led.history, "zz"))
check("balances untouched by rejected calls", led.balance("a") == 100 and led.balance("b") == 0)
check("transfer to unknown account raises KeyError", raises(KeyError, led.transfer, "a", "zz", 10))
check("failed transfer is atomic", led.balance("a") == 100 and led.history("a") == [])
check("transfer to self rejected", raises(ValueError, led.transfer, "a", "a", 10))
check("overdraft transfer rejected", raises(ValueError, led.transfer, "a", "b", 101))
led.transfer("a", "b", 40)
check("transfer moves money", led.balance("a") == 60 and led.balance("b") == 40)
check("transfer history kinds", led.history("a") == [("transfer_out", 40)] and led.history("b") == [("transfer_in", 40)])
h = led.history("a"); h.append(("x", 1))
check("history is a copy", led.history("a") == [("transfer_out", 40)])
led.open("c", 199); led.open("d", 0)
led.apply_interest(150)
check("interest rounds down", led.balance("c") == 201 and led.history("c") == [("interest", 2)])
check("zero interest leaves no entry", led.balance("d") == 0 and led.history("d") == [])
check("small balance interest", led.balance("a") == 60 and led.balance("b") == 40)
finish()
''',
    {"ledger.py": r'''def _amount(cents):
    if isinstance(cents, bool) or not isinstance(cents, int) or cents <= 0:
        raise ValueError("amount must be a positive int")


class Ledger:
    def __init__(self):
        self.accounts = {}
        self.log = {}

    def open(self, name, cents=0):
        if name in self.accounts:
            raise ValueError("account exists")
        if isinstance(cents, bool) or not isinstance(cents, int) or cents < 0:
            raise ValueError("bad opening amount")
        self.accounts[name] = cents
        self.log[name] = []

    def balance(self, name):
        return self.accounts[name]

    def deposit(self, name, cents):
        _amount(cents)
        self.accounts[name] += cents
        self.log[name].append(("deposit", cents))

    def withdraw(self, name, cents):
        _amount(cents)
        if self.accounts[name] < cents:
            raise ValueError("insufficient funds")
        self.accounts[name] -= cents
        self.log[name].append(("withdraw", cents))

    def transfer(self, src, dst, cents):
        _amount(cents)
        for name in (src, dst):
            if name not in self.accounts:
                raise KeyError(name)
        if src == dst:
            raise ValueError("same account")
        if self.accounts[src] < cents:
            raise ValueError("insufficient funds")
        self.accounts[src] -= cents
        self.accounts[dst] += cents
        self.log[src].append(("transfer_out", cents))
        self.log[dst].append(("transfer_in", cents))

    def history(self, name):
        return list(self.log[name])

    def apply_interest(self, rate_bps):
        for name in self.accounts:
            amount = self.accounts[name] * rate_bps // 10000
            if amount > 0:
                self.accounts[name] += amount
                self.log[name].append(("interest", amount))
'''},
)

# ----------------------------------------------------------------------------- justify
_JUSTIFY_REF = r'''def justify(text, width):
    if width < 1:
        raise ValueError("width must be at least 1")
    lines, cur, cur_len = [], [], 0
    for word in text.split():
        if cur and cur_len + 1 + len(word) > width:
            lines.append(cur)
            cur, cur_len = [], 0
        cur_len = cur_len + 1 + len(word) if cur else len(word)
        cur.append(word)
    if cur:
        lines.append(cur)
    out = []
    for i, words in enumerate(lines):
        if len(words) == 1 or i == len(lines) - 1:
            out.append(" ".join(words).ljust(width))
            continue
        gaps = len(words) - 1
        space, extra = divmod(width - sum(len(w) for w in words), gaps)
        line = ""
        for j, word in enumerate(words[:-1]):
            line += word + " " * (space + (1 if j < extra else 0))
        out.append(line + words[-1])
    return out
'''

_add(
    "justify", "library",
    "Implement justify(text, width) in /work/justify.py as described in SPEC.md. Public tests: "
    "python -m unittest -q test_public",
    {
        "SPEC.md": r'''# Text justification

Create `justify.py` with one function:

    justify(text, width) -> list of str

It breaks `text` into lines of exactly `width` characters.

## Rules
1. Words are the pieces of `text` separated by any whitespace (spaces, tabs,
   newlines). Empty or whitespace-only text gives `[]`.
2. Pack words greedily: put as many words on a line as fit, with at least one
   space between words.
3. A line with two or more words that is not the last line is fully justified:
   the spaces are spread between the words as evenly as possible, and if they do
   not divide evenly the gaps on the left get one more space than the gaps on
   the right.
4. A line that holds a single word is left-justified and padded with spaces on
   the right up to `width`.
5. The last line is left-justified, with single spaces between words, and padded
   with spaces on the right up to `width`.
6. A word longer than `width` goes on a line of its own. It is neither broken
   nor padded, so that line is longer than `width`.
7. `width` below 1 raises `ValueError`.
''',
        "test_public.py": r'''import unittest
from justify import justify


class PublicTests(unittest.TestCase):
    def test_example(self):
        text = "This is an example of text justification."
        self.assertEqual(justify(text, 16), ["This    is    an", "example  of text", "justification.  "])
''',
    },
    _JUSTIFY_REF.replace("def justify", "def _ref") + r'''
from justify import justify
check("classic example", justify("This is an example of text justification.", 16)
      == ["This    is    an", "example  of text", "justification.  "])
cases = [
    ("What must be acknowledgment shall be", 16),
    ("Science is what we understand well enough to explain to a computer. Art is everything else we do", 20),
    ("a b c d e f g", 5),
    ("one\ttwo\n\nthree   four", 9),
    ("tiny supercalifragilistic word here", 6),
    ("single", 10),
    ("left gaps get the extra spaces first always", 13),
]
for text, width in cases:
    check(f"case width={width} text={text[:18]!r}", justify(text, width) == _ref(text, width))
check("long word not padded", justify("tiny supercalifragilistic word", 6)[1] == "supercalifragilistic")
check("single word line padded", justify("abc defghij k", 7)[0] == "abc    ")
check("empty text", justify("", 5) == [] and justify("  \n ", 5) == [])
check("bad width", raises(ValueError, justify, "a", 0))
finish()
''',
    {"justify.py": _JUSTIFY_REF},
)

# ----------------------------------------------------------------------------- semver
_add(
    "semver", "library",
    "Implement semver.py in /work (functions compare, sort_versions, bump) as described in SPEC.md. "
    "Public tests: python -m unittest -q test_public",
    {
        "SPEC.md": r'''# Semantic version ordering

Create `semver.py` with three functions:

    compare(a, b)        -> -1, 0 or 1
    sort_versions(items) -> new list, ascending
    bump(version, part)  -> str

## Rules
1. A version is `MAJOR.MINOR.PATCH`, optionally followed by `-PRERELEASE` and
   then optionally by `+BUILD`, as in Semantic Versioning 2.0.0. The three
   numbers have no leading zeros. PRERELEASE and BUILD are dot-separated
   identifiers; every identifier is non-empty and uses only `0-9A-Za-z-`.
2. Every function raises `ValueError` for an invalid version. These are invalid:
   `"1.2"`, `"01.2.3"`, `"1.2.3-"`, `"1.2.3-01"` (a numeric prerelease identifier
   with a leading zero), `"1.2.3-a..b"`, `"v1.2.3"`, `"1.2.3 "`, `"1.2.3\n"`.
3. `compare` orders by MAJOR, then MINOR, then PATCH, as numbers. BUILD is
   ignored completely: `compare("1.0.0+a", "1.0.0+b")` is 0.
4. A version with a prerelease is lower than the same version without one:
   `1.0.0-rc.1 < 1.0.0`.
5. Two prereleases are compared identifier by identifier from the left. Numeric
   identifiers compare as numbers. A numeric identifier is lower than an
   alphanumeric one. Alphanumeric identifiers compare in ASCII order. If all
   shared identifiers are equal, the version with more identifiers is higher:
   `1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-alpha.beta < 1.0.0-beta.2 < 1.0.0-beta.11 < 1.0.0-rc.1`.
6. `sort_versions` returns a new list in ascending order and does not change its
   argument. Versions that compare equal keep their original relative order.
7. `bump(version, part)` increases `"major"`, `"minor"` or `"patch"` and resets
   the lower parts to 0. The result never has a prerelease or build part:
   `bump("1.2.3-rc.1+b7", "patch")` is `"1.2.4"`, `bump("1.2.3", "minor")` is
   `"1.3.0"`. Any other `part` raises `ValueError`.
''',
        "test_public.py": r'''import unittest
import semver


class PublicTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(semver.compare("1.2.3", "1.10.0"), -1)
        self.assertEqual(semver.compare("2.0.0", "2.0.0"), 0)

    def test_bump(self):
        self.assertEqual(semver.bump("1.2.3", "minor"), "1.3.0")
''',
    },
    r'''
import semver
chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2",
         "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0", "1.0.1", "1.2.0", "2.0.0"]
ok = all(semver.compare(chain[i], chain[j]) == (-1 if i < j else (1 if i > j else 0))
         for i in range(len(chain)) for j in range(len(chain)))
check("full precedence chain", ok)
check("numbers compared numerically", semver.compare("1.2.3", "1.10.0") == -1 and semver.compare("10.0.0", "9.9.9") == 1)
check("build ignored", semver.compare("1.0.0+a", "1.0.0+b") == 0 and semver.compare("1.0.0-rc.1+x", "1.0.0-rc.1") == 0)
check("numeric below alphanumeric", semver.compare("1.0.0-1", "1.0.0-a") == -1)
check("hyphen identifiers allowed", semver.compare("1.0.0-x-y.1", "1.0.0-x-y.2") == -1)
for bad in ("1.2", "01.2.3", "1.2.3-", "1.2.3-01", "1.2.3-a..b", "v1.2.3", "1.2.3 ", "1.2.3\n", "", "1.2.3+", "1.2.3-a_b"):
    check(f"invalid {bad!r} rejected", raises(ValueError, semver.compare, bad, "1.0.0"))
check("zero prerelease id is valid", semver.compare("1.0.0-0", "1.0.0-1") == -1)
src = ["1.0.0", "1.0.0-rc.1", "0.9.0", "1.0.0+build5", "1.0.0-alpha"]
res = semver.sort_versions(src)
check("sort order and stability", res == ["0.9.0", "1.0.0-alpha", "1.0.0-rc.1", "1.0.0", "1.0.0+build5"])
check("sort does not mutate", src == ["1.0.0", "1.0.0-rc.1", "0.9.0", "1.0.0+build5", "1.0.0-alpha"] and res is not src)
check("sort rejects invalid", raises(ValueError, semver.sort_versions, ["1.0.0", "nope"]))
check("bump", semver.bump("1.2.3-rc.1+b7", "patch") == "1.2.4" and semver.bump("1.2.3", "minor") == "1.3.0"
      and semver.bump("1.2.3", "major") == "2.0.0")
check("bump validates", raises(ValueError, semver.bump, "1.2.3", "build") and raises(ValueError, semver.bump, "1.2", "major"))
finish()
''',
    {"semver.py": r'''import functools
import re

_ID = r"[0-9A-Za-z-]+"
_RE = re.compile(rf"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-({_ID}(?:\.{_ID})*))?(?:\+({_ID}(?:\.{_ID})*))?", re.ASCII)


def _parse(version):
    match = _RE.fullmatch(version) if isinstance(version, str) else None
    if not match:
        raise ValueError(f"invalid version: {version!r}")
    pre = match.group(4).split(".") if match.group(4) else []
    for ident in pre:
        if ident.isdigit() and len(ident) > 1 and ident[0] == "0":
            raise ValueError(f"invalid version: {version!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))), pre


def _key(ident):
    return (0, int(ident), "") if ident.isdigit() else (1, 0, ident)


def compare(a, b):
    (na, pa), (nb, pb) = _parse(a), _parse(b)
    if na != nb:
        return -1 if na < nb else 1
    if not pa and not pb:
        return 0
    if not pa or not pb:
        return 1 if not pa else -1
    ka, kb = [_key(i) for i in pa], [_key(i) for i in pb]
    return 0 if ka == kb else (-1 if ka < kb else 1)


def sort_versions(items):
    return sorted(items, key=functools.cmp_to_key(compare))


def bump(version, part):
    (major, minor, patch), _ = _parse(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown part: {part!r}")
'''},
)

# ----------------------------------------------------------------------------- rle
_RLE_REF = r'''_SPECIAL = "0123456789\\"


def _unit(c):
    return "\\" + c if c in _SPECIAL else c


def encode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        n = j - i
        out.append(str(n) + _unit(s[i]) if n >= 3 else _unit(s[i]) * n)
        i = j
    return "".join(out)


def decode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] in "0123456789":
            j += 1
        count = 1
        if j > i:
            digits = s[i:j]
            if digits[0] == "0" or int(digits) < 3:
                raise ValueError("bad count")
            count = int(digits)
        if j >= len(s):
            raise ValueError("count without a character")
        c = s[j]
        if c == "\\":
            if j + 1 >= len(s) or s[j + 1] not in _SPECIAL:
                raise ValueError("bad escape")
            j += 1
            c = s[j]
        out.append(c * count)
        i = j + 1
    return "".join(out)
'''

_add(
    "rle", "library",
    "Implement encode and decode in /work/rle.py as described in SPEC.md.",
    {
        "SPEC.md": r'''# Run-length encoding with escapes

Create `rle.py` with two functions, `encode(s) -> str` and `decode(s) -> str`.

## Rules
1. A run of 3 or more equal characters is written as the run length in decimal
   followed by the character: `"aaaa"` becomes `"4a"`. Runs of 1 or 2 characters
   are written as they are: `"aab"` stays `"aab"`.
2. The ASCII digits `0`-`9` and the backslash are special. Whenever one of them
   is written to the output as a character, it is preceded by a backslash.
   So `"7"` becomes `\7`, a single backslash becomes two backslashes, `"1111"`
   becomes `4\1`, and `"22"` becomes `\2\2`.
3. `encode("")` is `""`.
4. `decode` reverses `encode`. It also accepts input that is well formed but not
   minimal, for example `"aaa"` (decodes to `"aaa"`).
5. `decode` raises `ValueError` for malformed input:
   - a count that is not followed by a character, such as `"12"`;
   - a count below 3 or a count with a leading zero, such as `"2a"` or `"03a"`;
   - a backslash at the very end;
   - a backslash followed by anything other than a digit or a backslash, such as `\a`.
6. `decode(encode(s)) == s` must hold for every string `s`, including strings
   that contain digits, backslashes, spaces and newlines.
''',
    },
    _RLE_REF.replace("def encode", "def _enc").replace("def decode", "def _dec") + r'''
import random
import rle
check("basic run", rle.encode("aaaa") == "4a" and rle.encode("aab") == "aab")
check("digits escaped", rle.encode("7") == "\\7" and rle.encode("22") == "\\2\\2" and rle.encode("1111") == "4\\1")
check("backslash escaped", rle.encode("\\") == "\\\\" and rle.encode("\\\\\\") == "3\\\\")
check("long run", rle.encode("x" * 12 + "y") == "12xy")
check("empty", rle.encode("") == "" and rle.decode("") == "")
rng = random.Random(7)
ok_enc = ok_rt = True
for _ in range(300):
    s = "".join(rng.choice("ab01\\ \n") for _ in range(rng.randint(0, 14)))
    if rle.encode(s) != _enc(s):
        ok_enc = False
    if rle.decode(rle.encode(s)) != s:
        ok_rt = False
check("encode matches the specification on random strings", ok_enc)
check("round trip on random strings", ok_rt)
check("decode non-minimal input", rle.decode("aaa") == "aaa" and rle.decode("3ab") == "aaab")
check("decode escapes", rle.decode("4\\1\\\\") == "1111\\")
for bad in ("12", "2a", "03a", "a\\", "\\a", "5", "1a"):
    check(f"decode rejects {bad!r}", raises(ValueError, rle.decode, bad))
finish()
''',
    {"rle.py": _RLE_REF},
)

# ----------------------------------------------------------------------------- logsum
_LOG_SAMPLE = """2024-05-01T12:00:03Z GET /api/users 200 123ms
2024-05-01T12:00:04Z POST /api/users 201 310ms
2024-05-01T12:00:05Z GET /api/items 500 87ms
this line is broken
2024-05-01T12:00:07Z GET /api/users 200 97ms
2024-05-01T12:00:09Z GET /api/items 200 45ms
2024-05-01T12:00:11Z GET /health 204 2ms
"""

_add(
    "logsum", "script",
    "Write summarize.py in /work so that `python summarize.py LOGFILE` prints the JSON summary "
    "described in SPEC.md. A sample log is access.log.",
    {
        "SPEC.md": r'''# Access log summary

Write `summarize.py`. It is run as

    python summarize.py LOGFILE

and prints one JSON object to stdout. `access.log` is a sample. The grader runs
your script on other log files that follow the same rules.

## Line format
    TIMESTAMP METHOD PATH STATUS LATENCY
for example `2024-05-01T12:00:03Z GET /api/users 200 123ms`. Fields are separated
by single spaces.

## Output
    {"total": n, "malformed": m, "by_class": {...}, "error_rate": r, "slowest": [...]}

## Rules
1. Blank lines are ignored: they are neither valid nor malformed.
2. A line is malformed when it does not have exactly 5 fields, when METHOD is not
   one of GET, POST, PUT, PATCH, DELETE, when STATUS is not an integer from 100
   to 599, or when LATENCY is not a non-negative integer directly followed by
   `ms`. Malformed lines are counted in `malformed` and otherwise skipped.
3. `total` is the number of valid lines.
4. `by_class` maps `"1xx"` ... `"5xx"` to the number of valid lines with a status
   in that class. Classes that do not occur are left out.
5. `error_rate` is the share of valid lines with a 5xx status, rounded to 4
   decimal places. It is 0.0 when there are no valid lines.
6. `slowest` lists at most 3 paths as `{"path": p, "mean_ms": x, "count": c}`.
   Only paths with at least 2 valid requests qualify. They are ordered by mean
   latency, highest first; paths with exactly the same mean are ordered by path
   in ascending order. `mean_ms` is rounded to 1 decimal place (order by the
   unrounded mean).
7. Different methods on the same path belong to the same path.
8. If LOGFILE does not exist, print nothing to stdout, write a message to stderr
   and exit with code 2.
''',
        "access.log": _LOG_SAMPLE,
    },
    r'''
import pathlib
tmp = pathlib.Path("/tmp/verify_log"); tmp.mkdir(parents=True, exist_ok=True)
lines = [
    "2024-05-01T00:00:01Z GET /a 200 100ms", "2024-05-01T00:00:02Z POST /a 201 201ms",
    "2024-05-01T00:00:03Z GET /b 500 300ms", "2024-05-01T00:00:04Z GET /b 503 1ms",
    "", "2024-05-01T00:00:05Z GET /c 404 150ms", "2024-05-01T00:00:06Z DELETE /c 204 151ms",
    "2024-05-01T00:00:07Z GET /d 200 900ms", "2024-05-01T00:00:08Z FETCH /d 200 900ms",
    "2024-05-01T00:00:09Z GET /e 200 150ms", "2024-05-01T00:00:10Z GET /e 301 151ms",
    "2024-05-01T00:00:11Z GET /f 99 10ms", "2024-05-01T00:00:12Z GET /f 200 10", "garbage",
    "2024-05-01T00:00:13Z GET /f 200 -5ms", "2024-05-01T00:00:14Z PUT /g 100 5ms extra",
    "2024-05-01T00:00:15Z PATCH /z 200 1ms", "2024-05-01T00:00:16Z PATCH /z 200 2ms",
]
(tmp / "t.log").write_text("\n".join(lines) + "\n")
r = sh(f"python summarize.py {tmp}/t.log")
check("exit code 0", r.returncode == 0)
try:
    out = json.loads(r.stdout)
except Exception:
    out = {}
check("total counts valid lines", out.get("total") == 11)
check("malformed counted, blank ignored", out.get("malformed") == 6)
check("by_class", out.get("by_class") == {"2xx": 7, "3xx": 1, "4xx": 1, "5xx": 2})
check("error_rate rounded to 4 places", isinstance(out.get("error_rate"), float) and abs(out["error_rate"] - 0.1818) < 1e-9)
check("slowest ordering, tie-break and min count", out.get("slowest") == [
    {"path": "/a", "mean_ms": 150.5, "count": 2}, {"path": "/b", "mean_ms": 150.5, "count": 2},
    {"path": "/c", "mean_ms": 150.5, "count": 2}])
(tmp / "empty.log").write_text("\n\n")
r = sh(f"python summarize.py {tmp}/empty.log")
try:
    out = json.loads(r.stdout)
except Exception:
    out = {}
check("empty log", out == {"total": 0, "malformed": 0, "by_class": {}, "error_rate": 0.0, "slowest": []})
r = sh(f"python summarize.py {tmp}/missing.log")
check("missing file: code 2, stderr only", r.returncode == 2 and r.stdout.strip() == "" and r.stderr.strip() != "")
finish()
''',
    {"summarize.py": r'''import json
import os
import sys

METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def parse(line):
    parts = line.split(" ")
    if len(parts) != 5:
        return None
    _, method, path, status, latency = parts
    if method not in METHODS or not status.isdigit() or not 100 <= int(status) <= 599:
        return None
    if not latency.endswith("ms") or not latency[:-2].isdigit():
        return None
    return path, int(status), int(latency[:-2])


def main():
    name = sys.argv[1]
    if not os.path.isfile(name):
        print(f"summarize: {name}: no such file", file=sys.stderr)
        sys.exit(2)
    total = malformed = errors = 0
    classes, paths = {}, {}
    with open(name) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            rec = parse(line)
            if rec is None:
                malformed += 1
                continue
            path, status, ms = rec
            total += 1
            key = f"{status // 100}xx"
            classes[key] = classes.get(key, 0) + 1
            errors += status >= 500
            paths.setdefault(path, []).append(ms)
    ranked = sorted(((sum(v) / len(v), p, len(v)) for p, v in paths.items() if len(v) >= 2),
                    key=lambda t: (-t[0], t[1]))[:3]
    print(json.dumps({
        "total": total, "malformed": malformed, "by_class": dict(sorted(classes.items())),
        "error_rate": round(errors / total, 4) if total else 0.0,
        "slowest": [{"path": p, "mean_ms": round(m, 1), "count": c} for m, p, c in ranked],
    }))


main()
'''},
)

# ----------------------------------------------------------------------------- workdays
_add(
    "workdays", "library",
    "Implement workdays and add_workdays in /work/workdays.py as described in SPEC.md. "
    "A sample holiday file is holidays.txt.",
    {
        "SPEC.md": r'''# Working days

Create `workdays.py` with two functions:

    workdays(start, end, holidays_path)     -> int
    add_workdays(start, n, holidays_path)   -> str

Dates are ISO strings such as `"2024-05-01"`. `holidays_path` is the path of a
text file that lists holidays (`holidays.txt` is a sample).

## Rules
1. A working day is a Monday to Friday that is not listed as a holiday.
2. `workdays` counts the working days d with `start <= d < end`: the start date
   is included and the end date is not.
3. If `end` is before `start`, the result is the negative of
   `workdays(end, start, holidays_path)`.
4. `add_workdays(start, n, ...)` returns the date reached by moving forward n
   working days from `start`, as an ISO string. The start date itself is never
   counted, and it may be a weekend or a holiday. With n = 0 the start date is
   returned unchanged. Negative n raises `ValueError`.
5. Holiday file format: one ISO date per line. A `#` starts a comment that runs
   to the end of the line, so a line may be a comment only or a date followed by
   a comment. Blank lines are ignored. Surrounding spaces are ignored.
6. A holiday that falls on a weekend changes nothing. A date listed twice counts
   once.
7. A date string that is not a valid ISO date raises `ValueError`.
''',
        "holidays.txt": "# public holidays\n2024-05-01  # Labour Day\n\n2024-05-09\n2024-05-20\n",
    },
    r'''
import pathlib
import workdays as wd
hol = "/tmp/verify_holidays.txt"
pathlib.Path(hol).write_text("# test\n2024-07-04 # Thu\n\n  2024-07-06  \n2024-07-04\n#2024-07-08\n2024-07-15\n")
check("plain week", wd.workdays("2024-07-08", "2024-07-15", hol) == 5)
check("end exclusive, start inclusive", wd.workdays("2024-07-01", "2024-07-02", hol) == 1 and wd.workdays("2024-07-01", "2024-07-01", hol) == 0)
check("holiday removed once", wd.workdays("2024-07-01", "2024-07-08", hol) == 4)
check("weekend holiday ignored, commented date ignored", wd.workdays("2024-07-05", "2024-07-10", hol) == 3)
check("reversed range negative", wd.workdays("2024-07-08", "2024-07-01", hol) == -4)
check("add zero", wd.add_workdays("2024-07-06", 0, hol) == "2024-07-06")
check("add from weekend", wd.add_workdays("2024-07-06", 1, hol) == "2024-07-08")
check("add skips holiday", wd.add_workdays("2024-07-03", 1, hol) == "2024-07-05" and wd.add_workdays("2024-07-12", 1, hol) == "2024-07-16")
check("add many", wd.add_workdays("2024-07-01", 10, hol) == "2024-07-17")
check("negative n", raises(ValueError, wd.add_workdays, "2024-07-01", -1, hol))
check("invalid date", raises(ValueError, wd.workdays, "2024-13-01", "2024-07-01", hol) and raises(ValueError, wd.add_workdays, "July 1", 1, hol))
finish()
''',
    {"workdays.py": r'''from datetime import date, timedelta


def _holidays(path):
    days = set()
    with open(path) as f:
        for line in f:
            text = line.split("#", 1)[0].strip()
            if text:
                days.add(date.fromisoformat(text))
    return days


def _working(day, holidays):
    return day.weekday() < 5 and day not in holidays


def workdays(start, end, holidays_path):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if last < first:
        return -workdays(end, start, holidays_path)
    holidays = _holidays(holidays_path)
    count, day = 0, first
    while day < last:
        count += _working(day, holidays)
        day += timedelta(days=1)
    return count


def add_workdays(start, n, holidays_path):
    if n < 0:
        raise ValueError("n must not be negative")
    holidays = _holidays(holidays_path)
    day = date.fromisoformat(start)
    while n > 0:
        day += timedelta(days=1)
        if _working(day, holidays):
            n -= 1
    return day.isoformat()
'''},
)

# ----------------------------------------------------------------------------- calc
_add(
    "calc", "library",
    "Implement evaluate(expr) in /work/calc.py as described in SPEC.md. Public tests: "
    "python -m unittest -q test_public",
    {
        "SPEC.md": r'''# Integer calculator

Create `calc.py` with one function:

    evaluate(expr: str) -> int

It evaluates an arithmetic expression over integers.

## Rules
1. Tokens: non-negative integer literals, the operators `+ - * / % **`, and
   parentheses. Spaces and tabs between tokens are ignored.
2. Precedence, lowest to highest: `+ -`, then `* / %`, then unary minus, then
   `**`. So `-2 ** 2` is `-4` and `2 * -3` is `-6`.
3. `+ - * / %` are left-associative. `**` is right-associative:
   `2 ** 3 ** 2` is `512`. The exponent may itself start with a unary minus.
4. `/` is integer division that truncates toward zero: `7 / -2` is `-3` and
   `-7 / 2` is `-3`. (Python's `//` rounds down instead, which is different.)
5. `%` is the remainder that goes with that division, so its sign follows the
   left operand: `-7 % 3` is `-1` and `7 % -3` is `1`.
6. Division or remainder by zero raises `ZeroDivisionError`.
7. A negative exponent raises `ValueError`.
8. Anything that is not a valid expression raises `ValueError`: empty input,
   unbalanced parentheses, a missing operand (`"1 +"`), two numbers in a row
   (`"1 2"`), a decimal point (`"1.5"`), letters, or a unary plus (`"+1"`).
9. Results must be exact for large numbers: `evaluate("10 ** 30 / 7")` is
   `142857142857142857142857142857`.
''',
        "test_public.py": r'''import unittest
from calc import evaluate


class PublicTests(unittest.TestCase):
    def test_precedence(self):
        self.assertEqual(evaluate("1 + 2 * 3"), 7)
        self.assertEqual(evaluate("(1 + 2) * 3"), 9)

    def test_error(self):
        with self.assertRaises(ValueError):
            evaluate("1 +")
''',
    },
    r'''
from calc import evaluate
cases = {
    "1 + 2 * 3": 7, "(1 + 2) * 3": 9, "10 - 4 - 3": 3, "2 ** 3 ** 2": 512, "-2 ** 2": -4,
    "2 * -3": -6, "7 / -2": -3, "-7 / 2": -3, "7 / 2": 3, "-7 % 3": -1, "7 % -3": 1, "7 % 3": 1,
    "2 ** -0": 1, "--3": 3, "-(2 + 3) * 2": -10, "\t12*( 3+4 )%5 ": 4, "100 / 10 / 5": 2,
    "2 ** 2 ** 0": 2, "(((7)))": 7, "10 ** 30 / 7": 142857142857142857142857142857,
    "-(10 ** 20) / 3": -33333333333333333333,
}
for expr, want in cases.items():
    try:
        got = evaluate(expr)
    except Exception as exc:
        got = repr(exc)
    check(f"evaluate({expr!r}) == {want}", got == want and isinstance(got, int))
for bad in ("", "   ", "(1 + 2", "1 + 2)", "1 +", "1 2", "1.5", "a + 1", "+1", "2 ** -1", "()", "1 * / 2", "2 *** 2"):
    check(f"evaluate({bad!r}) raises ValueError", raises(ValueError, evaluate, bad))
for bad in ("1 / 0", "5 % (2 - 2)"):
    check(f"evaluate({bad!r}) raises ZeroDivisionError", raises(ZeroDivisionError, evaluate, bad))
finish()
''',
    {"calc.py": r'''def _tokens(expr):
    out, i = [], 0
    while i < len(expr):
        c = expr[i]
        if c in " \t":
            i += 1
        elif c in "0123456789":
            j = i
            while j < len(expr) and expr[j] in "0123456789":
                j += 1
            out.append(int(expr[i:j]))
            i = j
        elif expr.startswith("**", i):
            out.append("**")
            i += 2
        elif c in "+-*/%()":
            out.append(c)
            i += 1
        else:
            raise ValueError(f"unexpected character {c!r}")
    return out


class _Parser:
    def __init__(self, tokens):
        self.tokens, self.pos = tokens, 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self):
        token = self.peek()
        self.pos += 1
        return token

    def expr(self):
        value = self.term()
        while self.peek() in ("+", "-"):
            value = value + self.term() if self.take() == "+" else value - self.term()
        return value

    def term(self):
        value = self.unary()
        while self.peek() in ("*", "/", "%"):
            op = self.take()
            right = self.unary()
            if op == "*":
                value *= right
                continue
            if right == 0:
                raise ZeroDivisionError("division by zero")
            quotient = abs(value) // abs(right)
            if (value < 0) != (right < 0):
                quotient = -quotient
            value = quotient if op == "/" else value - right * quotient
        return value

    def unary(self):
        if self.peek() == "-":
            self.take()
            return -self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek() == "**":
            self.take()
            exponent = self.unary()
            if exponent < 0:
                raise ValueError("negative exponent")
            return base ** exponent
        return base

    def atom(self):
        token = self.take()
        if isinstance(token, int):
            return token
        if token == "(":
            value = self.expr()
            if self.take() != ")":
                raise ValueError("missing closing parenthesis")
            return value
        raise ValueError("expected a number")


def evaluate(expr):
    parser = _Parser(_tokens(expr))
    value = parser.expr()
    if parser.pos != len(parser.tokens):
        raise ValueError("unexpected trailing input")
    return value
'''},
)

# ----------------------------------------------------------------------------- merge_records
_add(
    "merge_records", "library",
    "Implement merge(a, b) in /work/merge_records.py as described in SPEC.md. Public tests: "
    "python -m unittest -q test_public",
    {
        "SPEC.md": r'''# Merging record lists

Create `merge_records.py` with one function:

    merge(a, b) -> list of dict

`a` and `b` are lists of records. A record is a dict with an `"id"`, an
`"updated"` timestamp (an ISO string; compare timestamps as plain strings), and
any other fields.

## Rules
1. The result has one record per id that occurs in `a` or `b`, sorted by id in
   ascending order.
2. An id that occurs in only one list: its record is copied to the result.
3. An id that occurs in both lists, with different timestamps: the record with
   the later `"updated"` wins as a whole. Fields that exist only in the older
   record are dropped (except tags, see rule 5).
4. An id that occurs in both lists with the same timestamp: the records are
   combined field by field. If a field is missing or `None` on one side, the
   other side's value is used. If both sides have a value, the value from `b`
   wins.
5. `"tags"` is special. Whenever a result record has tags, they are the union of
   the tags of both versions of that record, de-duplicated and sorted. This also
   applies under rule 3: tags of the older record are kept. If neither version
   has a `"tags"` field, the result has none.
6. A record without an `"id"` raises `ValueError`. The same id occurring twice
   within one list raises `ValueError`.
7. Neither input list, nor any record or tag list inside them, may be modified.
   Result records are new objects.
''',
        "test_public.py": r'''import unittest
from merge_records import merge


class PublicTests(unittest.TestCase):
    def test_newer_wins(self):
        a = [{"id": 1, "updated": "2024-01-01", "name": "old"}]
        b = [{"id": 1, "updated": "2024-02-01", "name": "new"}]
        self.assertEqual(merge(a, b), [{"id": 1, "updated": "2024-02-01", "name": "new"}])

    def test_union_of_ids(self):
        a = [{"id": 2, "updated": "2024-01-01"}]
        b = [{"id": 1, "updated": "2024-01-01"}]
        self.assertEqual([r["id"] for r in merge(a, b)], [1, 2])
''',
    },
    r'''
import copy
from merge_records import merge
a = [
    {"id": 3, "updated": "2024-03-01", "name": "c-old", "email": "c@x", "tags": ["z", "a"]},
    {"id": 1, "updated": "2024-01-05", "name": "a", "phone": None, "tags": ["m"]},
    {"id": 5, "updated": "2024-01-01", "name": "only-a", "tags": ["b", "a", "b"]},
    {"id": 7, "updated": "2024-06-01", "name": "newer-in-a", "note": "keep"},
]
b = [
    {"id": 1, "updated": "2024-01-05", "name": "b", "phone": "555", "city": None, "tags": ["k", "m"]},
    {"id": 3, "updated": "2024-04-01", "name": "c-new", "tags": ["q"]},
    {"id": 4, "updated": "2024-02-02", "name": "only-b"},
    {"id": 7, "updated": "2024-05-01", "name": "older-in-b", "extra": 1, "tags": ["t"]},
]
a0, b0 = copy.deepcopy(a), copy.deepcopy(b)
res = merge(a, b)
by = {r.get("id"): r for r in res}
check("sorted union of ids", [r.get("id") for r in res] == [1, 3, 4, 5, 7])
check("same timestamp merged field-wise, b wins conflicts",
      by.get(1) == {"id": 1, "updated": "2024-01-05", "name": "b", "phone": "555", "city": None, "tags": ["k", "m"]})
check("newer record wins whole, tags unioned",
      by.get(3) == {"id": 3, "updated": "2024-04-01", "name": "c-new", "tags": ["a", "q", "z"]})
check("only in b copied", by.get(4) == {"id": 4, "updated": "2024-02-02", "name": "only-b"})
check("only in a copied, tags sorted and de-duplicated",
      by.get(5) == {"id": 5, "updated": "2024-01-01", "name": "only-a", "tags": ["a", "b"]})
check("newer in a wins and keeps older tags",
      by.get(7) == {"id": 7, "updated": "2024-06-01", "name": "newer-in-a", "note": "keep", "tags": ["t"]})
check("inputs not modified", a == a0 and b == b0)
check("results are new objects", all(r is not x for r in res for x in a + b))
check("missing id rejected", raises(ValueError, merge, [{"updated": "2024-01-01"}], []))
check("duplicate id rejected", raises(ValueError, merge, [], [{"id": 1, "updated": "x"}, {"id": 1, "updated": "y"}]))
check("empty", merge([], []) == [])
finish()
''',
    {"merge_records.py": r'''import copy


def _index(records):
    out = {}
    for rec in records:
        if "id" not in rec:
            raise ValueError("record without id")
        if rec["id"] in out:
            raise ValueError(f"duplicate id {rec['id']!r}")
        out[rec["id"]] = rec
    return out


def merge(a, b):
    ia, ib = _index(a), _index(b)
    result = []
    for key in sorted(set(ia) | set(ib)):
        ra, rb = ia.get(key), ib.get(key)
        if ra is None or rb is None:
            rec = copy.deepcopy(ra if rb is None else rb)
            versions = [rec]
        else:
            if ra["updated"] > rb["updated"]:
                rec = copy.deepcopy(ra)
            elif rb["updated"] > ra["updated"]:
                rec = copy.deepcopy(rb)
            else:
                rec = {}
                for field in list(ra) + [f for f in rb if f not in ra]:
                    va, vb = ra.get(field), rb.get(field)
                    rec[field] = copy.deepcopy(va if vb is None else vb)
            versions = [ra, rb]
        if any("tags" in v for v in versions):
            tags = set()
            for v in versions:
                tags |= set(v.get("tags") or [])
            rec["tags"] = sorted(tags)
        result.append(rec)
    return result
'''},
)
