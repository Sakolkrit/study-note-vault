---
tags: [programming, python, testing, tooling]
status: growing
level: 4
tier: 2
---

# pytest

**pytest** is a Python tool that finds your test functions, runs them all, and tells you clearly which ones passed and which failed, and why.

## 🟢 1. Simple (age 12)
A test is a tiny question you ask your own code: "If I add 2 and 3, do I get 5?" pytest asks all your questions in one go and shows you a scoreboard. If your code breaks later, the scoreboard tells you straight away which question went wrong.

**Example:** you build a calculator. You write ten little questions like "2 + 3 should be 5" and "10 - 4 should be 6". Every time you change the calculator, you press one button, and all ten questions are checked in a second.

## 🔵 2. How it works
1. Put your code in a file, for example `calc.py`, with a function `add(a, b)` ([[Python]]).
2. Make a second file whose name starts with `test_`, for example `test_calc.py`.
3. In it, write small functions whose names start with `test_`. Inside each, use a plain `assert`: `assert add(2, 3) == 5`.
4. Run `pytest` in the [[Command Line]]. It looks for every `test_*.py` file and every `test_*` function by itself.
5. Each test either passes (a dot) or fails (an F). A failing `assert` is shown with the real values, so you see what went wrong without adding print lines.
6. All tests run even if one fails, so one report shows everything that is broken.
7. Install it inside a project's own environment ([[Virtual Environments]]): `pip install pytest`.
8. **The catch:** a test only checks what you thought to ask. Passing tests do not prove the code is right, and tests that are hard to run get skipped.

## 🟠 3. Technical detail
- **Discovery:** files named `test_*.py` or `*_test.py`, functions named `test_*` (the default rules; from memory).
- **Plain `assert`:** pytest rewrites the assert so that a failure prints the values on both sides (in the output below: `assert 0.30000000000000004 == 0.3`).
- **Fixtures:** functions that prepare something a test needs (a temporary file, a database, a small data set) and hand it to the test by name (from memory).
- **Parametrised tests:** one test function run over many inputs with a decorator, so each input is reported separately (from memory).
- **Floating-point numbers:** `0.1 + 0.2 == 0.3` is false in floating point, a classic surprise. Compare with `pytest.approx(0.3)` or `math.isclose` (shown in the code below). This matters for numeric code such as [[NumPy]] and machine-learning work.
- **Exit code:** non-zero when any test fails, so build systems and [[Git]] hooks can stop a bad change.
- **In machine learning:** tests are used to check data loaders, shapes of arrays and the maths of small functions, so that a silent bug (such as a wrong [[Matrix]] shape) is caught early. They do not replace the evaluation of a model ([[Evaluation Metrics]]).
- **Alternatives:** the standard library's `unittest`, which needs a class and special `self.assert...` methods.

```python
import os, subprocess, sys, tempfile, textwrap
try: import pytest
except ImportError: sys.exit("pip install pytest first")
d = tempfile.mkdtemp()
body = textwrap.dedent('''
    def add(a, b): return a + b
    def test_small(): assert add(2, 3) == 5
    def test_negative(): assert add(-1, -1) == -2
    def test_wrong(): assert add(0.1, 0.2) == 0.3          # a real beginner surprise: floating point
    def test_more(): assert add(10, 5) == 15
''')
open(os.path.join(d, "test_add.py"), "w").write(body)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", d], capture_output=True, text=True, cwd=d)
lines = r.stdout.strip().splitlines()
print("pytest summary:", lines[-1])
print("what it shows for the failure:", [l.strip() for l in lines if l.strip().startswith("E ")][:2])
# the same checks as a plain script with asserts: it stops at the first failure
ns = {}; exec("def add(a, b): return a + b", ns); checks = [(2, 3, 5), (-1, -1, -2), (0.1, 0.2, 0.3), (10, 5, 15)]; done = 0
try:
    for a, b, want in checks: assert ns["add"](a, b) == want; done += 1
except AssertionError: pass
print(f"plain script with assert: stops after {done} of {len(checks)} checks; the 4th check never ran")
import math
print("0.1 + 0.2 == pytest.approx(0.3):", 0.1 + 0.2 == pytest.approx(0.3), "; math.isclose(0.1 + 0.2, 0.3):", math.isclose(0.1 + 0.2, 0.3))
ut = textwrap.dedent('''
    import unittest
    def add(a, b): return a + b
    class T(unittest.TestCase):
        def test_small(self): self.assertEqual(add(2, 3), 5)
        def test_negative(self): self.assertEqual(add(-1, -1), -2)
        def test_wrong(self): self.assertEqual(add(0.1, 0.2), 0.3)
        def test_more(self): self.assertEqual(add(10, 5), 15)
    if __name__ == "__main__": unittest.main()
''').strip()
open(os.path.join(d, "ut_add.py"), "w").write(ut)
r2 = subprocess.run([sys.executable, "ut_add.py"], capture_output=True, text=True, cwd=d)
print("same four checks with the standard unittest module:", r2.stderr.strip().splitlines()[-1], "| lines in the test file: unittest", len(ut.splitlines()), "vs pytest", len(body.strip().splitlines()))
```

## 🔴 4. Deep / formal
**What a test is.** A test is a small program with a clear pass or fail result, written so it can run without a human. A good test has one reason to fail, runs fast, does not depend on other tests, and names what it checks.

**Trade-offs:**
- **Few, fast, focused tests** are run often. Slow tests get skipped and then stop protecting you.
- **Tests that copy the code's own logic** will pass even when the logic is wrong. Check against values worked out another way, such as a hand calculation (see the notes' **Example** blocks, which are computed in Python first for the same reason).
- **Random behaviour** needs a fixed seed, or the test fails some days and not others.
- **Coverage tools** (from memory) count which lines the tests run. High coverage shows lines were run, not that they were checked.

**Limits:**
- Tests show the presence of bugs, never their absence.
- A test that passes for the wrong reason (an assert that cannot fail) is worse than no test.

**What came before, and how this improved on it:**
- **Before:** checking by hand with `print` and by eye, or a script of plain `assert` lines, or the standard `unittest` module (modelled on the Java JUnit; from memory).
- **Its limit:** a plain assert script stops at the first failure, so you learn about one problem at a time. `unittest` needs a class and special methods for every check.
- **What changed:** pytest finds plain functions by name, uses the ordinary `assert`, runs every test, and reports the values behind each failure. It began as `py.test` in the mid-2000s, led by Holger Krekel (from memory, not checked).
- **How much better:** in the code below, four checks written as a plain script stop after 2 checks when the third fails, so the fourth never runs. pytest reports "1 failed, 3 passed" and shows `assert 0.30000000000000004 == 0.3`. The same four checks in `unittest` take 8 lines against 5 for pytest. The failing check is itself the lesson: `0.1 + 0.2 == 0.3` is false, and `pytest.approx` fixes it.

**Check it yourself:** run the code in level 3 above (it needs `pip install pytest`). It prints every number quoted in this block.

## Related
- [[Python]] · [[Command Line]] · [[Virtual Environments]] · [[Git]] · [[NumPy]] · [[Jupyter Notebook]] · [[Evaluation Metrics]] · [[Matrix]] · [[FastAPI]] · [[Streamlit]] · [[Iris Classifier Service]]

## Sources
- pytest documentation, https://docs.pytest.org (discovery rules, fixtures, parametrisation and `approx`: from memory, not reopened this session).
- The code and every number in the history block were run for this note with pytest 8.4.2 in a fresh virtual environment (Python 3.9, Windows).
- No image: I found no free, on-topic picture for the tool itself on Wikimedia Commons (only a logo), so I added none.
- Names and dates marked "from memory" are not checked.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/dev-tools pytest:::Asks all your tiny questions about your own code in one go and shows a scoreboard, so a later break tells you straight away which question failed.

#flashcards/programming/concept #flashcards/topic/dev-tools How do you write a test?::Put code in a file such as `calc.py`. In a second file named `test_calc.py`, write functions whose names start with `test_` and use a plain `assert` on a comparison, for example that `add(2, 3)` equals 5.

#flashcards/programming/concept #flashcards/topic/dev-tools How does pytest find tests?::Run `pytest` in the command line. It looks for every `test_*.py` file (or `*_test.py`) and every `test_*` function by itself (default rules, from memory).

#flashcards/programming/concept #flashcards/topic/dev-tools What does a pass and a fail look like?::A pass shows a dot, a fail shows an F. A failing `assert` prints the real values on both sides, so you do not need print lines. All tests run even if one fails.

#flashcards/programming/concept #flashcards/topic/dev-tools How do you install pytest?::Inside the project's own environment: `pip install pytest`.

#flashcards/programming/reverse #flashcards/topic/dev-tools Fixture:::A function that prepares something a test needs (a temporary file, a database, a small data set) and hands it to the test by name (from memory).

#flashcards/programming/reverse #flashcards/topic/dev-tools Parametrised test:::One test function run over many inputs with a decorator, so each input is reported separately (from memory).

#flashcards/programming/example #flashcards/topic/dev-tools Why is adding 0.1 and 0.2 and comparing with 0.3 a classic surprise?::In floating point the sum is 0.30000000000000004, not exactly 0.3. Compare with `pytest.approx(0.3)` or `math.isclose`.

#flashcards/programming/concept #flashcards/topic/dev-tools What exit code does pytest give when a test fails, and why does it matter?::A non-zero code, so build systems and Git hooks can stop a bad change.

#flashcards/programming/concept #flashcards/topic/dev-tools What is the catch of tests?::A test only checks what you thought to ask. Passing tests do not prove the code is right, and tests that are hard to run get skipped.

#flashcards/programming/concept #flashcards/topic/dev-tools What is the alternative in the standard library?::`unittest`, which needs a class and special `self.assert...` methods.

#flashcards/programming/cloze #flashcards/topic/dev-tools In machine learning, tests check ==data loaders, array shapes and small maths functions==, so a silent bug such as a wrong matrix shape is caught early. They do not replace evaluating a model.
