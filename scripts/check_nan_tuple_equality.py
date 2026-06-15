import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_a = (1.0, 2.0, float("nan"), float("nan"))
_b = (1.0, 2.0, float("nan"), float("nan"))
print("fresh-NaN tuple equality:", _a == _b)
assert (_a == _b) is False