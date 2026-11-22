---
tags: [python, virtual-environment, venv, pip, packages, programming, tooling]
status: growing
level: 4
tier: 2
---

# Virtual Environments

A **virtual environment** is a private, separate copy of Python's package shelf for one project. Different projects often need different versions of the same library, and one shared shelf cannot hold two versions at once. With a virtual environment, each project gets its own shelf, so installing or upgrading things for one project cannot break another. It is the standard first step of any [[Python]] project, and it matters for the libraries in this vault, such as [[NumPy]], [[PyTorch]] and [[TensorFlow]], which can be picky about each other's versions.

## 🟢 1. Simple (age 12)
Imagine your family shares one big toolbox. You borrow a screwdriver of one size for a model aeroplane. Your brother needs the *same-named* screwdriver but in a different size for his bike. If there is only one slot for it in the box, you keep swapping, and each swap breaks the other's job. The fix: each of you gets your own small toolbox. Now you can each have exactly the tools, and the versions, you need, and nobody's work gets spoilt.

**Example:** Project 1 needs an old version of a library to run. Project 2 needs the newest. With one shared set, installing the new one breaks Project 1. With two virtual environments, each project keeps the version it wants.

## 🔵 2. How it works
1. **Python has a shelf of installed packages.** When you write `import numpy`, Python looks on that shelf. The tool `pip` puts new packages on it.
2. **A virtual environment is a folder.** Inside is its own copy (really a small launcher) of Python and its own empty shelf. You make one with a single command, `python -m venv env`, where `env` is just a folder name you choose.
3. **You switch it on ("activate").** Activating puts the environment's folder first on your `PATH` (the list of places the shell looks for programs, see [[Command Line]]). After that, the plain command `python` means *this environment's* Python.
4. **Install what the project needs.** `pip install something` now installs into the environment's shelf only. Your normal Python is left alone.
5. **Write down what you used.** `pip freeze` lists every package and its exact version. Saving that list in a file (`requirements.txt`) lets anyone, including you in a year, rebuild the same environment with `pip install -r requirements.txt`.
6. **Throw it away when done.** Because an environment is just a folder, deleting the folder deletes it. The project's own code lives elsewhere and is not touched.

![[dependency-hell-diagram.svg]]
*The problem that environments solve, in a diagram with Polish labels: "program A" and "program B" both need ("wymagania", requirements) program C, but A needs version v0.1 and B needs version v0.2, which gives a "konflikt zależności" (dependency conflict). The translation of the three words is mine, from my own knowledge of Polish, and the file's own description ("a diagram showing badly defined dependencies") agrees with it. One shared shelf can hold only one version of C. Two environments can each hold their own. Image: Lukas3 (Polish Wikipedia), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Pieklo_zaleznosci.svg), public domain.*

## 🟠 3. Technical detail
- **Making one:** `python -m venv env` (the standard library's `venv`, no install needed). The code made one: the folder holds `Include`, `Lib`, `Scripts` and a text file `pyvenv.cfg` (on Linux and macOS the folder is called `bin` instead of `Scripts`, from memory; I ran only on Windows).
- **The config file:** `pyvenv.cfg` has three lines in this Python (3.9.6): `home` (where the real Python is), `include-system-site-packages` (`false` by default) and `version`.
- **Starting almost empty:** the new environment held only `pip` and `setuptools`. In the code, `import numpy` worked in the normal Python and failed in the environment with `ModuleNotFoundError`. So installing a library globally does **not** make it available inside an environment.
- **Sharing the global packages on purpose:** `python -m venv --system-site-packages env` lets the environment see the global ones as well. In the code, numpy was importable again. This is convenient but gives up the isolation, so a clash with the global version becomes possible.
- **Telling you are inside one:** in an environment `sys.prefix` differs from `sys.base_prefix`. The code printed `True` inside and `False` in the normal Python.
- **Installing:** the code built a tiny package (`tinydemo`) and installed it into the environment from a local folder with no internet (using `--no-index --no-build-isolation --no-deps`). The environment could import it and the normal Python could not. `pip freeze` then showed `tinydemo==0.1`.
- **Activation scripts:** the `Scripts` folder had `activate` (for Git Bash and similar), `activate.bat` (for the older Windows `cmd`) and `Activate.ps1` (for PowerShell). I did not run these scripts. Their job, from memory, is to put the folder first on `PATH` and change your prompt, and `deactivate` undoes it. What I did check is the effect: with the environment folder first on `PATH`, the name `python` is found inside the environment, and without it, it is not. I also did not need activation at all in the code, since I ran the environment's Python by its full path, which works the same way.
- **A Windows detail I ran into:** my first version of this check started `python` with a changed `PATH` from inside Python, and it still found the normal Python. On Windows, a program launched by Python looks the bare name up on the *launching* program's `PATH`, not the new one (that explanation is from memory). The final code uses `shutil.which` with the new `PATH` instead, which shows the lookup directly.
- **Version rules:** requirements are written as ranges. `>=1.2,<2` allows 1.5 and rejects 2.0, and `~=1.4.2` ("compatible release") allows 1.4.9 and rejects 1.5.0. The code tests these with the `packaging` library (version 23.1, the library `pip` uses for this, from memory).
- **Why conflicts are hard:** the number of possible version combinations is a product (equation 1 below). The code enumerates them for a small example, finds the valid ones, and shows that program A (needs C below 0.2) and B (needs C at or above 0.2) cannot share one environment.
- **Disposable:** the code deleted the environment folder and the environment was gone, with the normal Python untouched.
- **Other tools** (from memory; I did not run any): `virtualenv` (an older, separate tool, version 20.7.2 is installed here), `conda` environments, `poetry` and `uv`. They solve the same problem with extra features such as installing other Python versions or locking exact versions.
- **Limits:** the code ran on Windows with Python 3.9.6, no internet, and used one tiny local package. I did not install any large package from the internet, run an activation script, or test other platforms.

The code covers six things: (1) making an environment and its contents, (2) isolation from the global packages, and the option that removes it, (3) installing a package only into the environment, (4) what activation changes, (5) version rules and a small conflict, (6) deleting the environment.

```python
import subprocess, sys, os, tempfile, shutil, itertools
from packaging.specifiers import SpecifierSet
from packaging.version import Version
D = tempfile.mkdtemp(prefix='vedemo'); BIN = 'Scripts' if os.name == 'nt' else 'bin'
def run(args, **kw): return subprocess.run(args, capture_output=True, text=True, **kw)
def make(name, *flags):
    r = run([sys.executable, '-m', 'venv', *flags, os.path.join(D, name)]); assert r.returncode == 0, r.stderr
    return os.path.join(D, name, BIN, 'python.exe' if os.name == 'nt' else 'python')
def py(exe, code): return run([exe, '-c', code]).stdout.strip()
# 1) make an environment: it is just a folder
env = make('env')
print('top-level entries:', sorted(os.listdir(os.path.join(D, 'env'))))
cfg = dict(l.split(' = ') for l in open(os.path.join(D, 'env', 'pyvenv.cfg')).read().split('\n') if ' = ' in l)
print('pyvenv.cfg keys:', sorted(cfg), '| include-system-site-packages =', cfg['include-system-site-packages'], '| activation scripts:', sorted(f for f in os.listdir(os.path.join(D, 'env', BIN)) if f.lower().startswith('activate')))
# 2) isolation: the environment starts nearly empty and cannot see the global packages
print('inside the environment, sys.prefix differs from sys.base_prefix:', py(env, 'import sys; print(sys.prefix != sys.base_prefix)'), '| in the normal Python:', py(sys.executable, 'import sys; print(sys.prefix != sys.base_prefix)'))
pkgs = run([env, '-m', 'pip', 'list', '--format=freeze']).stdout.split(); print('packages in the new environment:', sorted(p.split('==')[0] for p in pkgs))
print('import numpy in the normal Python:', py(sys.executable, 'import numpy; print("works")'), '| in the environment:', run([env, '-c', 'import numpy']).stderr.strip().split('\n')[-1])
shared = make('shared', '--system-site-packages'); print('with --system-site-packages the environment sees numpy again:', py(shared, 'import numpy; print("works")'))
# 3) install a tiny package of our own (no internet needed) and see that only the environment gets it
pk = os.path.join(D, 'tinydemo'); os.makedirs(pk)
open(os.path.join(pk, 'setup.py'), 'w').write("from setuptools import setup\nsetup(name='tinydemo', version='0.1', py_modules=['tinydemo'])\n")
open(os.path.join(pk, 'tinydemo.py'), 'w').write("def hello(): return 'hi from tinydemo'\n")
r = run([env, '-m', 'pip', 'install', '-q', '--no-index', '--no-build-isolation', '--no-deps', pk]); print('install into the environment succeeded:', r.returncode == 0)
print('environment imports it:', py(env, 'import tinydemo; print(tinydemo.hello())'), '| normal Python can import it:', run([sys.executable, '-c', 'import tinydemo']).returncode == 0)
print('pip freeze in the environment:', [l for l in run([env, '-m', 'pip', 'freeze']).stdout.split() if l.startswith('tinydemo')])
# 4) what "activate" does: put the environment's folder first on PATH, so the name python finds its python
NEWPATH = os.path.join(D, 'env', BIN) + os.pathsep + os.environ['PATH']
found_new, found_old = shutil.which('python', path=NEWPATH), shutil.which('python', path=os.environ['PATH'])
print('the name "python" with the environment folder first on PATH is found inside the environment:', os.path.commonpath([found_new, os.path.join(D, 'env')]) == os.path.join(D, 'env'), '| and uses it:', py(found_new, 'import sys; print(sys.prefix != sys.base_prefix)'), '| with PATH unchanged it is found inside the environment:', os.path.commonpath([found_old, D]) == D)
# 5) why separate environments: two programs may need different versions of the same library
print('version rules: >=1.2,<2 allows 1.5:', Version('1.5') in SpecifierSet('>=1.2,<2'), '| allows 2.0:', Version('2.0') in SpecifierSet('>=1.2,<2'), '| ~=1.4.2 allows 1.4.9:', Version('1.4.9') in SpecifierSet('~=1.4.2'), '| allows 1.5.0:', Version('1.5.0') in SpecifierSet('~=1.4.2'))
need = {('A', '1'): SpecifierSet('<0.2'), ('A', '2'): SpecifierSet('>=0.2'), ('B', '1'): SpecifierSet('<0.2'), ('B', '2'): SpecifierSet('>=0.2'), ('B', '3'): SpecifierSet('>=0.2')}
A, B, C = ['1', '2'], ['1', '2', '3'], ['0.1', '0.2', '0.3']
allc = list(itertools.product(A, B, C)); ok = [(a, b, c) for a, b, c in allc if Version(c) in need[('A', a)] and Version(c) in need[('B', b)]]
print('combinations of versions:', len(allc), '(= 2 x 3 x 3) | combinations where every need is met:', len(ok), '| newest valid:', max(ok, key=lambda t: (int(t[0]), int(t[1]), Version(t[2]))))
print('program A needs C<0.2 and program B needs C>=0.2 -> versions that satisfy both:', [c for c in C if Version(c) in SpecifierSet('<0.2') and Version(c) in SpecifierSet('>=0.2')], '(none: that is the conflict; two environments fix it)')
# 6) an environment is disposable: delete the folder and it is gone
import time
for attempt in range(10):                                                   # on Windows a file can stay locked for a moment (for example while a scanner reads it), so retry
    try: shutil.rmtree(os.path.join(D, 'env')); break
    except PermissionError: time.sleep(1)
print('after deleting the folder the environment exists:', os.path.exists(os.path.join(D, 'env')), '| the normal Python is untouched:', py(sys.executable, 'print("still works")'))
shutil.rmtree(D, ignore_errors=True)
# top-level entries: ['Include', 'Lib', 'Scripts', 'pyvenv.cfg']
# pyvenv.cfg keys: ['home', 'include-system-site-packages', 'version'] | include-system-site-packages = false | activation scripts: ['Activate.ps1', 'activate', 'activate.bat']
# inside the environment, sys.prefix differs from sys.base_prefix: True | in the normal Python: False
# packages in the new environment: ['pip', 'setuptools']
# import numpy in the normal Python: works | in the environment: ModuleNotFoundError: No module named 'numpy'
# with --system-site-packages the environment sees numpy again: works
# install into the environment succeeded: True
# environment imports it: hi from tinydemo | normal Python can import it: False
# pip freeze in the environment: ['tinydemo==0.1']
# the name "python" with the environment folder first on PATH is found inside the environment: True | and uses it: True | with PATH unchanged it is found inside the environment: False
# version rules: >=1.2,<2 allows 1.5: True | allows 2.0: False | ~=1.4.2 allows 1.4.9: True | allows 1.5.0: False
# combinations of versions: 18 (= 2 x 3 x 3) | combinations where every need is met: 5 | newest valid: ('2', '3', '0.3')
# program A needs C<0.2 and program B needs C>=0.2 -> versions that satisfy both: [] (none: that is the conflict; two environments fix it)
# after deleting the folder the environment exists: False | the normal Python is untouched: still works
```
Reading the output (Python 3.9.6 on Windows, no internet):
- **Contents:** the folder holds `Include`, `Lib`, `Scripts` and `pyvenv.cfg`. The config has `home`, `include-system-site-packages` (set to `false`) and `version`, and the activation scripts are `Activate.ps1`, `activate` and `activate.bat`.
- **Isolation:** `sys.prefix` differs from `sys.base_prefix` inside the environment (`True`) and not in the normal Python (`False`). The new environment lists just `pip` and `setuptools`. `import numpy` works in the normal Python and fails in the environment with `No module named 'numpy'`. With `--system-site-packages` numpy is importable again.
- **Installing:** the install into the environment succeeded, the environment printed `hi from tinydemo`, the normal Python could not import it, and `pip freeze` listed `tinydemo==0.1`.
- **PATH:** with the environment's folder first on `PATH`, the name `python` is found inside the environment and it reports being in one. With `PATH` unchanged, it is not found inside the environment.
- **Versions:** `>=1.2,<2` allows 1.5 (`True`) and not 2.0 (`False`). `~=1.4.2` allows 1.4.9 (`True`) and not 1.5.0 (`False`).
- **Combinations:** 18 version combinations (2 × 3 × 3), 5 of which meet every need, and the newest valid one is A 2, B 3, C 0.3. For the A-against-B case, no version of C satisfies both rules, which is the conflict in the picture.
- **Delete:** after deleting the folder (with the retry loop) the environment does not exist and the normal Python still works.
- **Deleting can fail for a moment on Windows:** one of my runs stopped with `PermissionError ... being used by another process` on a cached file inside the environment, just after it had been used (the cause, such as a scanner reading the file, is my guess). The final code retries the delete up to 10 times, one second apart, and it worked.
- **Limits:** I ran the code on one computer. The whole script took about a minute (I did not time the individual steps), and the example numbers are tiny.

## 🔴 4. Deep / formal

**How many combinations?** Suppose a project needs $k$ libraries, and library $i$ comes in $n_i$ different versions. A resolver has to pick one version of each, so the number of ways to choose is:

$$N=\prod_{i=1}^{k} n_i$$

**Where:**
- $k$ is the number of libraries involved (including the libraries that your libraries need). **Matters:** every extra library multiplies the number of choices, so the count grows exponentially in $k$.
- $n_i$ is how many versions of library $i$ could be picked. **Matters:** old versions stay available, so $n_i$ grows over time as new releases come out.
- $\prod$ means "multiply them all together". **Matters:** it comes from the multiplication rule for independent choices: for each choice of the first library you can choose any version of the second, and so on.
- $N$ is the number of candidate sets, most of which are invalid because of version rules like `<0.2` and `>=0.2`. **Matters:** the resolver does not try them all one by one (that would be hopeless for large $k$). It prunes choices that already break a rule. In general, finding any valid set is believed to be a hard problem (from memory: it is NP-complete), but real projects usually resolve quickly. I did not test the speed of any real resolver.

**Example (tiny numbers):** library A with 2 versions, B with 3 versions, C with 3 versions.
1. $N=2\times3\times3=18$. The code enumerated 18 combinations.
2. The rules were: A 1 needs C below 0.2, A 2 needs C at or above 0.2, B 1 needs C below 0.2, and B 2 and B 3 need C at or above 0.2.
3. Counting by the version of C: C 0.1 allows A 1 with B 1 (1 set), C 0.2 allows A 2 with B 2 or B 3 (2 sets), C 0.3 allows A 2 with B 2 or B 3 (2 sets). The total is 5, and the code counted 5.
4. The newest valid choice is A 2, B 3 and C 0.3, as the code printed. Tools like `pip` usually try to pick the newest versions that work (from memory).
5. Meaning: even three libraries with a few versions each give 18 candidates, of which only 5 work. If a project asks for A at version 1 and B at version 2, the only C that fits both is none, and the answer is a conflict. A second environment, with its own C, avoids it.

**Other points (from memory, not checked here):**
- Environments are separate from containers (such as Docker) and from virtual machines: an environment isolates only Python packages, not the operating system or other programs. I did not check this against any documentation or test containers.
- Putting an environment folder inside a Git repository is usually a mistake because it is large and specific to one computer. The usual practice is to ignore it and keep only `requirements.txt`. I did not test this with [[Git]].
- Moving or copying an environment folder to another place or computer often breaks it, because `pyvenv.cfg` and the launchers record absolute paths. I did not test moving one.
- Installing a package with its binary parts (such as [[PyTorch]] or [[TensorFlow]]) downloads large files and may need a matching Python version. I did not install either here, since both are already installed globally.
- `pip` can also install straight from a Git address or a local folder (the code used a local folder). I did not test the Git form.

**What came before, and how this improved on it:**
- **Before:** install every package into one shared Python.
- **Its limit:** two projects that need different versions of the same package collide, and one project can break another.
- **What changed:** give each project its own folder of packages (`virtualenv` by Ian Bicking, 2007; `venv` built into Python from version 3.3 in 2012; both from memory, not checked).
- **How much better:** in the code below, the main Python on this machine has 577 packages installed, and a new environment has 2 (`pip` and `setuptools`) and does not see NumPy. Each project then adds only what it needs, at the version it needs.

**Check it yourself:**

```python
import os, subprocess, sys, tempfile
def pkgs(py): return [l.split()[0] for l in subprocess.run([py, "-m", "pip", "list", "--disable-pip-version-check"], capture_output=True, text=True).stdout.splitlines()[2:]]
glob = pkgs(sys.executable)
d = tempfile.mkdtemp(); subprocess.run([sys.executable, "-m", "venv", d], capture_output=True)
py = os.path.join(d, "Scripts" if os.name == "nt" else "bin", "python")
fresh = pkgs(py)
print("packages installed in the main Python:", len(glob), "(for example", ", ".join(glob[:3]) + ")")
print("packages in a brand-new environment:  ", len(fresh), "->", fresh)
print("numpy visible there:", "numpy" in fresh, "- each project installs only what it needs, at its own version")
```

## Related
[[Python]] · [[Command Line]] · [[Git]] · [[NumPy]] · [[Pandas]] · [[PyTorch]] · [[TensorFlow]] · [[Matplotlib]] · [[Recursion]]
- [[pytest]] · [[Jupyter Notebook]] · [[Docker]]
- [[MLOps]] · [[Model Monitoring Lab]]
## Sources
- Python documentation on the `venv` module and on `pip` (docs.python.org and pip.pypa.io), and the Python Packaging User Guide (named from memory; I did not open them for this note).
- Library versions in this note's run: Python 3.9.6, pip 24.3.1, setuptools 75.8.0 (global; the new environment received its own copy, whose version I did not print), packaging 23.1, virtualenv 20.7.2, found by importing them.
- Figure: [Pieklo_zaleznosci.svg](https://commons.wikimedia.org/wiki/File:Pieklo_zaleznosci.svg) (Lukas3 at Polish Wikipedia, public domain), on Wikimedia Commons. I viewed it before writing the caption.
- The code and every printed value were run for this note (Python 3.9.6 on Windows, no internet, in a temporary folder that the code deleted at the end). The whole script took about a minute; nothing else was timed.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/dev-tools Virtual environment:::Your own small toolbox per project. It is a folder with its own Python and its own empty shelf of packages, so projects do not break each other.

#flashcards/programming/concept #flashcards/topic/dev-tools Make a virtual environment and install into it.::`python -m venv env` makes it (`env` is just a folder name). After activating, `pip install something` goes into the environment's shelf only.

#flashcards/programming/concept #flashcards/topic/dev-tools What does activating do?::It puts the environment's folder first on `PATH`, so the plain command `python` means this environment's Python.

#flashcards/programming/concept #flashcards/topic/dev-tools How do you record and rebuild what a project used?::`pip freeze` lists every package with its exact version. Save it in `requirements.txt` and rebuild with `pip install -r requirements.txt`.

#flashcards/programming/concept #flashcards/topic/dev-tools How do you remove an environment?::Delete the folder. The project's own code lives elsewhere and is untouched. On Windows deleting can fail for a moment with a `PermissionError` if a file was just used.

#flashcards/programming/concept #flashcards/topic/dev-tools What does the environment folder hold?::`Include`, `Lib`, `Scripts` (called `bin` on Linux and macOS) and a text file `pyvenv.cfg` with `home`, `include-system-site-packages` (default `false`) and `version`.

#flashcards/programming/concept #flashcards/topic/dev-tools How does a new environment start and what does `--system-site-packages` do?::It starts almost empty, holding only `pip` and `setuptools`, so `import numpy` fails with `ModuleNotFoundError` even if NumPy is installed globally. `python -m venv --system-site-packages env` lets it see the global packages too, but gives up the isolation.

#flashcards/programming/concept #flashcards/topic/dev-tools How can you tell you are inside an environment?::`sys.prefix` differs from `sys.base_prefix`. The note's code printed `True` inside and `False` in the normal Python.

#flashcards/programming/concept #flashcards/topic/dev-tools Name the three activation scripts.::`activate` (Git Bash and similar), `activate.bat` (older Windows `cmd`) and `Activate.ps1` (PowerShell). The note did not run them.

#flashcards/programming/example #flashcards/topic/dev-tools What do `>=1.2,<2` and `~=1.4.2` allow?::`>=1.2,<2` allows 1.5 and rejects 2.0. `~=1.4.2` (compatible release) allows 1.4.9 and rejects 1.5.0.

#flashcards/programming/formula #flashcards/topic/dev-tools Write the number of possible version combinations and explain each symbol.
?
$$N=\prod_{i=1}^{k} n_i$$
- $k$: the number of libraries involved (including the libraries your libraries need). Every extra library multiplies the number of choices, so the count grows exponentially in $k$.
- $n_i$: how many versions of library $i$ could be picked. Old versions stay available, so $n_i$ grows as new releases come out.
- $\prod$: multiply them all together, by the multiplication rule for independent choices.

#flashcards/programming/example #flashcards/topic/dev-tools Library A has 2 versions, B has 3, C has 3. How many combinations, and how many meet all the rules?::$N=2\times3\times3=18$ combinations (the code enumerated 18). With the note's rules, 5 meet every need: C 0.1 allows 1 set, and C 0.2 and C 0.3 allow 2 each. The newest valid one is A 2, B 3, C 0.3.

#flashcards/programming/image #flashcards/topic/dev-tools ![[dependency-hell-diagram.svg]]
What does this picture show?
?
The problem environments solve, in a diagram with Polish labels: program A and program B both need program C, but A needs version v0.1 and B needs v0.2, which gives a dependency conflict. The note's translation of the three Polish words is its own, and the file's description agrees. One shared shelf cannot hold both versions. (From the note's caption, Wikimedia Commons.)

#flashcards/programming/cloze #flashcards/topic/dev-tools A virtual environment isolates ==Python libraries only==. A container isolates the whole file system.
