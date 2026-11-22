---
tags: [git, version-control, programming, tooling, command-line]
status: growing
level: 4
tier: 2
---

# Git

Git is a tool that **remembers every version of your files**. You tell it when to save a snapshot (called a *commit*), and later you can look at any old snapshot, compare two of them, undo a mistake, or try a risky idea on a separate line of work (a *branch*) without touching the main one. Nearly all programmers use it, and it works well for notes like the ones in this vault too. It is run from the command line, mostly with the word `git` followed by a command (see [[Command Line]]).

## 🟢 1. Simple (age 12)
Think of a video game with save points. Before a hard level you save. If you do badly, you load the save and try again. Git is a set of save points for your files, but better: every save point has a label, a date and a note saying what changed, and you can keep as many as you like. You can even say "let me try something wild on a copy" and, if it works, merge the copy back in.

**Example:** You are writing a story with a friend. You save a version called "first draft". You rewrite the ending and save "new ending". You then decide the old ending was better. Instead of retyping it, you ask Git for the file as it was in "first draft".

## 🔵 2. How it works
1. **Your files live in three places.** The **working folder** is the normal folder you edit. The **staging area** is a waiting room where you put the changes you want in the next snapshot. The **repository** is the hidden history of all snapshots (in a folder called `.git`).
2. **You choose what goes in a snapshot.** `git add file` moves the current version of a file into the staging area. `git commit -m "message"` then saves everything in the staging area as a new snapshot, with your message.
3. **Every snapshot has a unique ID.** It is a long code of letters and numbers (a *hash*), worked out from the contents. If anything changes, the ID changes (equation 1 below). Each snapshot also remembers which snapshot came before it, so the history is a chain.
4. **A branch is just a name that points at a snapshot.** Making a new branch is almost free. When you commit on a branch, its name moves forward to the new snapshot.
5. **Merging joins two lines of work.** If only one branch moved, Git just slides the other forward. If both moved, Git makes a new snapshot with two parents that combines them. If both changed the *same line*, Git cannot choose and asks you to decide: this is a *conflict*.
6. **Sharing uses a remote.** A *remote* is another copy of the repository (for example on a website). `git push` sends your new snapshots there, `git fetch` brings in the new ones from there, and `git pull` fetches and merges them.

![[git-operations.svg]]
*How common Git commands move changes between the places. Working Files (right): you edit here. `add` moves changes into the Stage. `commit` saves the Stage into a branch. `checkout` and `merge` bring a branch's files back into the working files. `push` sends branches to the Remote, `fetch` brings the remote's changes into your clone, and `pull` does a fetch and merge in one go. Image: Daniel Kinzler, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Git_operations.svg), CC BY 3.0. I did not check each arrow against Git's documentation beyond the explanations in this note.*

## 🟠 3. Technical detail
- **Commands used in the code** (each was run, in a throwaway repository, so nothing touched a real project): `git init` (start a repository), `git status --short`, `git add`, `git commit`, `git checkout -b name` (new branch), `git checkout name` (switch), `git merge`, `git revert` (undo a commit by adding a new one that cancels it), `git reset --hard` (move the branch back and throw the changes away), `git remote add`, `git push`, `git clone`, `git fetch`, `git pull`, `git rev-list --count`, `git hash-object`, `git cat-file -t` and `git diff --minimal --numstat`.
- **Status codes:** `??` means a new file that Git does not track yet, `A ` means staged, a blank line means everything is committed, ` M` means edited but not staged, and `UU` means a conflict in a file that both sides changed. The code prints each of these at the right moment.
- **IDs:** a file's id is the SHA-1 of a short header plus its contents (equation 1). Two different repositories that make a commit with the same content, message, author and date get the same commit id (the code sets a fixed author and date to show this, and both repositories gave `b73f66e…`). In normal use the date differs, so ids differ.
- **A branch is a tiny file:** `.git/refs/heads/main` held exactly 41 bytes in the code: 40 letters and numbers for the commit id, plus a newline. Moving a branch means rewriting this one small file.
- **Merging:** if the branch being merged is ahead of yours in a straight line, the merge is a *fast-forward* with no new commit (the code ends with 2 commits on `main`). If both branches have new commits, the merge creates a commit with **two parents** (the code shows 2 parents, and 5 commits in all).
- **Conflicts:** after a conflicting merge, the file contains markers: `<<<<<<< HEAD`, your version, `=======`, their version, `>>>>>>> branch name`. The code shows the exit code 1 and the file holding both versions. You edit the file to the final text, `git add` it, and commit.
- **Undoing:** `git revert` keeps history and adds a new commit that cancels the bad one (the code went from 4 to 5 commits). `git reset --hard HEAD~1` moves the branch back one commit and discards the changes in your files, which is dangerous with uncommitted work. In the code the undone commit **still existed** afterwards (`git cat-file -t` said `commit`) because Git keeps objects for a while, so it can often be recovered if you know its id. I did not test how long Git keeps such objects (from memory: weeks, until a clean-up runs).
- **Remote:** a local bare repository stood in for a website. After a push from one copy, a second clone was 1 commit behind after `fetch` and 0 behind after `pull`. I did not test any real network service, passwords or SSH keys.
- **Diffs:** the size of a diff in changed lines equals $n+m-2\,\mathrm{LCS}$ (equation 3). This is the same longest-common-subsequence idea as in [[Dynamic Programming]].
- **What I did not cover or run:** `.gitignore`, `git stash`, `git rebase`, tags, pull requests, hosting sites, signed commits, large files, or submodules. The code also does not time anything.
- **Limits:** the code ran Git 2.33.1 for Windows with the global and system settings switched off, a fixed author and date, and automatic line-ending conversion off. Other versions may print slightly different messages.

The code covers ten things: (1) a file's id by hand against `git hash-object`, (2) the three places, shown by `status`, (3) identical commits give identical ids, (4) a branch as a small file, (5) fast-forward and true merges, (6) a conflict and its resolution, (7) revert and reset, (8) push, fetch and pull with a local remote, (9) diff size against a longest-common-subsequence count, (10) short ids and the chance of a clash.

```python
import subprocess, os, tempfile, hashlib, random, math
ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_AUTHOR_NAME='Demo', GIT_AUTHOR_EMAIL='demo@example.com',
           GIT_COMMITTER_NAME='Demo', GIT_COMMITTER_EMAIL='demo@example.com', GIT_AUTHOR_DATE='2026-01-01T00:00:00+00:00', GIT_COMMITTER_DATE='2026-01-01T00:00:00+00:00')
def git(cwd, *a, ok=True):
    r = subprocess.run(['git', '-c', 'core.autocrlf=false', '-c', 'init.defaultBranch=main', *a], cwd=cwd, env=ENV, capture_output=True, text=True)
    if ok and r.returncode: raise RuntimeError(r.stderr)
    return r
def out(cwd, *a): return git(cwd, *a).stdout.strip()
def write(d, name, text): open(os.path.join(d, name), 'w', newline='\n').write(text)
def sha(x): return hashlib.sha1(x).hexdigest()
def repo():
    d = tempfile.mkdtemp(prefix='gitdemo'); git(d, 'init', '-q'); return d
# 1) a file's id is the SHA-1 of "blob <size>" + a zero byte + the content
d = repo(); write(d, 'a.txt', 'hello\n')
mine = sha(b'blob 6\x00hello\n'); theirs = out(d, 'hash-object', 'a.txt')
print('blob id by my own SHA-1:', mine, '| git hash-object:', theirs, '| equal:', mine == theirs)
other = sha(b'blob 6' + bytes([0]) + b'Hello' + bytes([10])); print('one capital letter changes the id completely:', other, '| bits that differ:', bin(int(mine, 16) ^ int(other, 16)).count('1'), 'of 160')
# 2) the three places: working folder, staging area, repository
print('new file       :', repr(git(d, 'status', '--short').stdout))
git(d, 'add', 'a.txt'); print('after add      :', repr(git(d, 'status', '--short').stdout))
git(d, 'commit', '-q', '-m', 'first'); print('after commit   :', repr(git(d, 'status', '--short').stdout))
write(d, 'a.txt', 'hello\nworld\n'); print('after an edit  :', repr(git(d, 'status', '--short').stdout))
git(d, 'checkout', '-q', '--', 'a.txt'); print('after checkout :', repr(open(os.path.join(d, 'a.txt')).read()), '(the edit is thrown away)')
# 3) same content and same details give the same commit id, anywhere
d2 = repo(); write(d2, 'a.txt', 'hello\n'); git(d2, 'add', 'a.txt'); git(d2, 'commit', '-q', '-m', 'first')
c1, c2 = out(d, 'rev-parse', 'HEAD'), out(d2, 'rev-parse', 'HEAD')
print('two separate repos, same file, message, author and date -> same commit id:', c1 == c2, '|', c1[:7])
# 4) a branch is only a small file holding a commit id
ref = open(os.path.join(d, '.git', 'refs', 'heads', 'main'), newline='').read()
print('branch file holds the commit id:', ref.strip() == c1, '| file size in bytes:', len(ref), '(40 hex letters + newline)')
# 5) branches and merging
git(d, 'checkout', '-q', '-b', 'feature'); write(d, 'b.txt', 'feature work\n'); git(d, 'add', '.'); git(d, 'commit', '-q', '-m', 'feature')
git(d, 'checkout', '-q', 'main'); git(d, 'merge', '-q', 'feature'); print('merge when main had not moved: fast-forward, commits on main =', out(d, 'rev-list', '--count', 'HEAD'))
git(d, 'checkout', '-q', '-b', 'side'); write(d, 'c.txt', 'side\n'); git(d, 'add', '.'); git(d, 'commit', '-q', '-m', 'side')
git(d, 'checkout', '-q', 'main'); write(d, 'd.txt', 'main\n'); git(d, 'add', '.'); git(d, 'commit', '-q', '-m', 'main moved')
git(d, 'merge', '-q', '--no-edit', 'side'); parents = out(d, 'rev-list', '--parents', '-n', '1', 'HEAD').split()[1:]
print('merge when both moved: a merge commit with', len(parents), 'parents | commits now', out(d, 'rev-list', '--count', 'HEAD'))
# 6) a conflict: both sides change the same line
k = repo(); write(k, 'f.txt', 'colour: red\n'); git(k, 'add', '.'); git(k, 'commit', '-q', '-m', 'base')
git(k, 'checkout', '-q', '-b', 'blue'); write(k, 'f.txt', 'colour: blue\n'); git(k, 'commit', '-q', '-am', 'blue')
git(k, 'checkout', '-q', 'main'); write(k, 'f.txt', 'colour: green\n'); git(k, 'commit', '-q', '-am', 'green')
r = git(k, 'merge', 'blue', ok=False); print('conflicting merge exit code:', r.returncode, '| status:', repr(git(k, 'status', '--short').stdout))
print('file now holds:', open(os.path.join(k, 'f.txt')).read().replace('\n', ' / '))
write(k, 'f.txt', 'colour: teal\n'); git(k, 'add', 'f.txt'); git(k, 'commit', '-q', '--no-edit'); print('after resolving by hand and committing, status is clean:', git(k, 'status', '--short').stdout == '')
# 7) undoing: revert adds a new commit, reset moves the branch back
n0 = out(k, 'rev-list', '--count', 'HEAD'); git(k, 'revert', '--no-edit', '-m', '1', 'HEAD'); n1 = out(k, 'rev-list', '--count', 'HEAD')
undone = out(k, 'rev-parse', 'HEAD'); git(k, 'reset', '-q', '--hard', 'HEAD~1')
print('commits before revert', n0, '| after revert', n1, '| after reset --hard HEAD~1', out(k, 'rev-list', '--count', 'HEAD'), '| the undone commit still exists:', out(k, 'cat-file', '-t', undone))
# 8) a remote: push, then a second copy fetches
bare = tempfile.mkdtemp(prefix='gitbare'); git(bare, 'init', '-q', '--bare')
git(d, 'remote', 'add', 'origin', bare); git(d, 'push', '-q', 'origin', 'main'); cl = tempfile.mkdtemp(prefix='gitclone'); git(cl, 'clone', '-q', bare, 'copy'); cp = os.path.join(cl, 'copy')
write(d, 'e.txt', 'later\n'); git(d, 'add', '.'); git(d, 'commit', '-q', '-m', 'later'); git(d, 'push', '-q', 'origin', 'main')
git(cp, 'fetch', '-q'); behind = out(cp, 'rev-list', '--count', 'HEAD..origin/main'); git(cp, 'pull', '-q')
print('clone is behind origin by', behind, 'commit(s) before pulling | after pull:', out(cp, 'rev-list', '--count', 'HEAD..origin/main'))
# 9) how big is a diff? lines removed + lines added = n + m - 2 * (longest common run of lines)
def lcs(a, b):
    t = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)): t[i + 1][j + 1] = t[i][j] + 1 if a[i] == b[j] else max(t[i][j + 1], t[i + 1][j])
    return t[-1][-1]
def diffsize(d, a, b):
    write(d, 'old.txt', '\n'.join(a) + '\n'); write(d, 'new.txt', '\n'.join(b) + '\n')
    s = git(d, 'diff', '--no-index', '--minimal', '--numstat', 'old.txt', 'new.txt', ok=False).stdout.split()
    return (int(s[0]), int(s[1])) if s else (0, 0)
A, B = list('abcdefgh'), list('abXdeYgh')
added, removed = diffsize(d, A, B)
print('git diff --minimal: added', added, 'removed', removed, '| n + m - 2*LCS =', len(A) + len(B) - 2 * lcs(A, B), '| LCS =', lcs(A, B))
rng = random.Random(1); same = 0
for _ in range(100):
    a = [rng.choice('abcd') for _ in range(rng.randint(1, 12))]; b = [rng.choice('abcd') for _ in range(rng.randint(1, 12))]
    same += sum(diffsize(d, a, b)) == len(a) + len(b) - 2 * lcs(a, b)
print('100 random line lists: git diff --minimal size equals n + m - 2*LCS in', same, 'of 100')
# 10) short ids: how likely is a clash among n objects when only b bits are shown?  p ~ 1 - exp(-n(n-1) / 2^(b+1))
p = lambda n, b: -math.expm1(-n * (n - 1) / 2 ** (b + 1))
print('7 hex letters = 28 bits, n=10000 objects: formula', round(p(10000, 28), 3), '| 160 bits (full id), n=10^9:', format(p(10 ** 9, 160), '.2e'))
hits = sum(len(set(rng.getrandbits(28) for _ in range(10000))) < 10000 for _ in range(2000)); print('simulation, 2000 trials of 10000 random 28-bit ids: clash in', hits / 2000)
def exact(n, b): return 1 - math.prod(1 - i / 2 ** b for i in range(n))
print('smaller case n=100 ids, b=10 bits (1024 values): formula', round(p(100, 10), 4), '| exact', round(exact(100, 10), 4))
# blob id by my own SHA-1: ce013625030ba8dba906f756967f9e9ca394464a | git hash-object: ce013625030ba8dba906f756967f9e9ca394464a | equal: True
# one capital letter changes the id completely: e965047ad7c57865823c7d992b1d046ea66edf78 | bits that differ: 84 of 160
# new file       : '?? a.txt\n'
# after add      : 'A  a.txt\n'
# after commit   : ''
# after an edit  : ' M a.txt\n'
# after checkout : 'hello\n' (the edit is thrown away)
# two separate repos, same file, message, author and date -> same commit id: True | b73f66e
# branch file holds the commit id: True | file size in bytes: 41 (40 hex letters + newline)
# merge when main had not moved: fast-forward, commits on main = 2
# merge when both moved: a merge commit with 2 parents | commits now 5
# conflicting merge exit code: 1 | status: 'UU f.txt\n'
# file now holds: <<<<<<< HEAD / colour: green / ======= / colour: blue / >>>>>>> blue / 
# after resolving by hand and committing, status is clean: True
# commits before revert 4 | after revert 5 | after reset --hard HEAD~1 4 | the undone commit still exists: commit
# clone is behind origin by 1 commit(s) before pulling | after pull: 0
# git diff --minimal: added 2 removed 2 | n + m - 2*LCS = 4 | LCS = 6
# 100 random line lists: git diff --minimal size equals n + m - 2*LCS in 100 of 100
# 7 hex letters = 28 bits, n=10000 objects: formula 0.17 | 160 bits (full id), n=10^9: 3.42e-31
# simulation, 2000 trials of 10000 random 28-bit ids: clash in 0.1715
# smaller case n=100 ids, b=10 bits (1024 values): formula 0.992 | exact 0.9933
```
Reading the output (Git 2.33.1 on Windows, Python 3.9):
- **Ids:** `hello` followed by a newline gets the id `ce013625…` by my own SHA-1 and by `git hash-object`. Changing the first letter to a capital gives `e965047a…`, where 84 of the 160 bits differ.
- **Three places:** `??` for a new file, `A ` after `add`, nothing after `commit`, ` M` after an edit, and the edit is thrown away by `checkout` (the file is back to `hello`).
- **Same ids:** two separate repositories with the same file, message, author and date give the same commit id, `b73f66e`.
- **Branch file:** it holds the commit id and is 41 bytes.
- **Merges:** the first merge was a fast-forward (2 commits on `main`). The second made a merge commit with 2 parents (5 commits in all).
- **Conflict:** exit code 1 and status `UU f.txt`. The file held `<<<<<<< HEAD / colour: green / ======= / colour: blue / >>>>>>> blue`. After editing and committing, the status was clean.
- **Undo:** 4 commits, then 5 after the revert, then 4 after the reset. The undone commit still existed.
- **Remote:** the clone was 1 commit behind before pulling and 0 after.
- **Diff:** for two 8-line files that differ in 2 lines, `git diff --minimal` reports 2 added and 2 removed, which equals $8+8-2\cdot6=4$. On 100 random line lists, the size equalled $n+m-2\,\mathrm{LCS}$ every time.
- **Short ids:** for 10,000 objects and 7 letters (28 bits) the formula gives 0.17, and a simulation of 2,000 trials gave 0.1715. For a full 160-bit id and a billion objects it gives about $3.4\times10^{-31}$. For 100 ids and 10 bits the formula gives 0.992 and the exact value is 0.9933.
- **Limits:** a throwaway local setup only, nothing timed.

## 🔴 4. Deep / formal

**What is a file's id?** Git names every stored file by a fingerprint. It builds a small header from the word `blob` and the file's size in bytes, adds a zero byte, then the file's contents, and takes the SHA-1 hash of the whole thing:

$$\text{id}=\mathrm{SHA1}\big(\texttt{"blob "}\,\Vert\,\text{size}\,\Vert\,\texttt{NUL}\,\Vert\,\text{content}\big)$$

**Where:**
- $\mathrm{SHA1}$ is a hash function: it turns any data into a 160-bit fingerprint (written as 40 hex letters and digits). **Matters:** the same input always gives the same fingerprint, and any tiny change gives a completely different one. This is what lets Git notice changes and check that nothing was corrupted.
- $\texttt{"blob "}$ and the size are a header that names the kind of object and its length. **Matters:** two objects of different kinds (a file, a tree of folders, a commit) with the same bytes still get different ids.
- $\texttt{NUL}$ is a single zero byte that ends the header. **Matters:** it separates the header from the content, so the two cannot be confused.
- $\text{content}$ is the file's exact bytes. **Matters:** the id depends on the content only, not on the file name or the date. Two identical files in different places share one stored copy.
- $\Vert$ means "joined together, one after the other". **Matters:** it shows the order: header first, then the content.

**Example (tiny numbers):** a file holding the text `hello` and a newline (6 bytes).
1. The header is `blob 6` followed by a zero byte, and then the 6 bytes of content: 13 bytes in all.
2. The SHA-1 of those 13 bytes is `ce013625030ba8dba906f756967f9e9ca394464a`. The code got the same value from `git hash-object`.
3. With a capital H (`Hello` and a newline) the id is `e965047ad7c57865823c7d992b1d046ea66edf78`, and 84 of the 160 bits are different.
4. Meaning: a one-letter change gives an unrelated id, so Git can tell at a glance that a file changed, and the id also works as a check that the data was not damaged.

**How likely is a clash of short ids?** Git often shows only the first few letters of an id (7 by default, from memory). Suppose $n$ objects have ids that behave like random $b$-bit numbers. The chance that at least two of them start the same way is, for $n$ much smaller than $2^{b}$:

$$p\ \approx\ 1-\exp\!\left(-\frac{n(n-1)}{2^{\,b+1}}\right)$$

**Where:**
- $n$ is the number of objects (commits, files, and so on). **Matters:** the number of pairs grows like $n^2/2$, which is why clashes show up much sooner than most people expect (the "birthday" effect).
- $b$ is the number of bits shown. **Matters:** each extra bit halves the chance of a clash. 7 hex letters is $7\times4=28$ bits.
- $n(n-1)/2$ is the number of pairs of objects. **Matters:** each pair clashes with probability $2^{-b}$, so the expected number of clashes is $n(n-1)/2^{b+1}$, which is the number inside the exponent.
- $1-\exp(-x)$ turns an expected count $x$ into a chance. **Matters:** for small $x$ it is about $x$, and it never goes above 1.
- The formula is an approximation. **Matters:** the code compared it with the exact value for a small case ($n=100$, $b=10$: 0.992 against 0.9933) and with a simulation for the case below, and it matched well. I did not prove the approximation.

**Example (tiny numbers):** $n=10{,}000$ objects, 7 letters ($b=28$).
1. Pairs: $n(n-1)=99{,}990{,}000$. $2^{29}=536{,}870{,}912$. The exponent is $99{,}990{,}000/536{,}870{,}912=0.1862$.
2. $p=1-e^{-0.1862}=0.170$. The code gave 0.17, and a simulation of 2,000 trials gave 0.1715.
3. For the full 160 bits and a billion objects the same formula gives about $3.4\times10^{-31}$.
4. Meaning: with 10,000 objects a 7-letter abbreviation clashes about one time in six, so Git makes the abbreviation longer as a repository grows. Full ids are, for all practical purposes, never going to clash by accident.

**How big is a change?** Compare an old file with $n$ lines and a new file with $m$ lines. The lines the two share, in the same order, form a *longest common subsequence* of length $\mathrm{LCS}$. Everything else is deleted from the old file or added to the new one, so the smallest possible diff has this many changed lines:

$$\#\text{changed lines}=\#\text{removed}+\#\text{added}=(n-\mathrm{LCS})+(m-\mathrm{LCS})=n+m-2\,\mathrm{LCS}$$

**Where:**
- $n$ and $m$ are the number of lines in the old and new file. **Matters:** they set the most that could change.
- $\mathrm{LCS}$ is the length of the longest run of lines that appear in both files in the same order (not necessarily next to each other). **Matters:** these are the lines Git can leave alone. The bigger it is, the smaller the diff.
- $n-\mathrm{LCS}$ is the number of old lines that have to be removed, and $m-\mathrm{LCS}$ the number of new lines that have to be added. **Matters:** a changed line counts twice, once as a removal and once as an addition.
- The result holds for the *smallest* diff made only of removals and additions. **Matters:** `git diff --minimal` asks for this. Without that flag, Git may use faster shortcuts that give a slightly bigger diff (from memory; I did not test the default).
- Finding the LCS is a dynamic-programming problem (see [[Dynamic Programming]], where the edit distance uses the same table), and I used that method in the code.

**Example (tiny numbers):** old lines `a b c d e f g h`, new lines `a b X d e Y g h`.
1. Lines in common, in order: `a b d e g h`, so $\mathrm{LCS}=6$. Here $n=m=8$.
2. Changed lines: $8+8-2\cdot6=4$. That is 2 removed (`c` and `f`) and 2 added (`X` and `Y`).
3. Git's `diff --minimal` reported 2 added and 2 removed, and on 100 random pairs of short line lists the size always matched the formula.
4. Meaning: two edited lines show up as 4 changed lines in a diff (2 minus lines and 2 plus lines), which is why a diff of a small edit can look bigger than the change felt.

**Other points (from memory, not checked here):**
- Git was created by Linus Torvalds in 2005 for managing the Linux kernel. I did not check the date or the story.
- SHA-1 is no longer considered safe against deliberate attacks, and Git has been moving toward SHA-256 (I did not check how far). For this note the point is only that the ids behave like fingerprints.
- Git's default diff algorithm and the way it stores history (it stores whole snapshots and packs them with deltas, not a list of line changes) are described from memory; I did not test them.
- The `main` against `master` default branch name differs between versions and settings. I set it explicitly.
- Hosting sites add pull requests, issues and permissions on top of Git. I did not use any.

**What came before, and how this improved on it:**
- **Before:** keep copies by hand, such as `report_v1`, `report_v2_final` and `report_v2_final_REAL`.
- **Its limit:** copies waste space, hide what changed, and make going back or combining work error-prone.
- **What changed:** store every version as a snapshot with a name and a message, and share the history. Linus Torvalds started Git in 2005 (from memory, not checked).
- **How much better:** in the code below, three versions of a 2,000-line file (one line changed each time) take 242,613 bytes as three raw copies and 15,762 bytes as three copies compressed alone. Git's loose objects take 16,402 bytes, no better than compressed copies, and after packing (`git gc`) 8,540 bytes, 28 times smaller than raw copies and 1.8 times smaller than compressed ones. All three versions stay recoverable, and going back two commits restored the original line. The real gain is the history, not only the space.

**Check it yourself:**

```python
import os, subprocess, tempfile
d = tempfile.mkdtemp()
g = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=d, capture_output=True, text=True).stdout.strip()
g("init", "-q")
lines = [f"line {i}: some text that stays the same" for i in range(2000)]
copies = 0
for v in range(1, 4):
    lines[v * 100] = f"line {v * 100}: CHANGED in version {v}"
    data = "\n".join(lines); open(os.path.join(d, "report.txt"), "w").write(data); copies += len(data)
    g("add", "report.txt"); g("commit", "-q", "-m", f"v{v}")
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(os.path.join(d, ".git", "objects")) for f in fs)
print("3 versions of a 2,000-line file, each with one line changed")
print("  three full copies (report_v1, report_v2, ...):", copies, "bytes")
import zlib
print("  the same three copies, each compressed alone:", 3 * len(zlib.compress(data.encode())), "bytes (fairer: git compresses too)")
print("  git history, loose objects:                   ", size, "bytes in .git/objects")
g("gc", "-q", "--aggressive")
size2 = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(os.path.join(d, ".git", "objects")) for f in fs)
print("  git history after packing (git gc):          ", size2, "bytes")
print("  versions recoverable:", g("log", "--oneline").count("\n") + 1)
g("checkout", "-q", "HEAD~2", "--", "report.txt"); print("  after going back two commits, line 300 reads:", open(os.path.join(d, "report.txt")).read().split("\n")[300])
```

## Related
[[Command Line]] · [[Python]] · [[Virtual Environments]] · [[Dynamic Programming]] · [[Trees]] · [[Graphs]] · [[Hash Tables]] · [[Big-O]] · [[Recursion]]
- [[pytest]] · [[Jupyter Notebook]] · [[Docker]]
- [[MLOps]] · [[Experiment Tracking]]
## Sources
- Git documentation (git-scm.com), on `status`, `commit`, `merge`, `revert`, `reset` and the object model (from memory; I did not open the pages for this note).
- Chacon and Straub, *Pro Git* (named from memory; I did not open it).
- Myers, E. W., "An O(ND) difference algorithm and its variations", 1986 (named from memory as the usual source for Git's diff; I did not open it, and I did not check which algorithm Git uses by default).
- Figure: [Git_operations.svg](https://commons.wikimedia.org/wiki/File:Git_operations.svg) (Daniel Kinzler, CC BY 3.0), on Wikimedia Commons. I viewed it before writing the caption.
- The code and every Example number were run for this note (Git 2.33.1 for Windows, Python 3.9, throwaway repositories in a temporary folder with the global and system settings switched off). Operations were counted, not timed.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/dev-tools Git:::Save points for your files. Every save point has a label, a date and a note saying what changed, and you can keep as many as you like.

#flashcards/programming/concept #flashcards/topic/dev-tools Name Git's three places for your files.::The working folder (where you edit), the staging area (a waiting room for the next snapshot) and the repository (the hidden history in `.git`).

#flashcards/programming/concept #flashcards/topic/dev-tools What do `git add file` and `git commit -m "message"` do?::`git add` moves the current version of a file into the staging area. `git commit -m` saves everything staged as a new snapshot with your message.

#flashcards/programming/concept #flashcards/topic/dev-tools What is a branch?::Just a name that points at a snapshot. Making one is almost free. A commit on the branch moves its name forward. The note's branch file `.git/refs/heads/main` was exactly 41 bytes: 40 characters of commit id plus a newline.

#flashcards/programming/concept #flashcards/topic/dev-tools Fast-forward merge against merge commit?::If the merged branch is ahead of yours in a straight line, Git slides forward with no new commit (fast-forward). If both branches moved, Git makes a new snapshot with two parents.

#flashcards/programming/concept #flashcards/topic/dev-tools What is a conflict and what does the file look like?::Both sides changed the same line and Git cannot choose. The file holds a `<<<<<<< HEAD` line, your version, a line of equals signs, their version, a `>>>>>>> branch name` line. You edit it, then commit. Status shows `UU`.

#flashcards/programming/concept #flashcards/topic/dev-tools `git push`, `git fetch`, `git pull`?::`push` sends your snapshots to the remote. `fetch` brings in the remote's new ones. `pull` does fetch and merge.

#flashcards/programming/example #flashcards/topic/dev-tools `git revert` against `git reset --hard HEAD~1`?::`revert` keeps history and adds a new commit that cancels the bad one (4 commits to 5 in the note's code). `reset --hard HEAD~1` moves the branch back one commit and discards the changes in your files (back to 4).

#flashcards/programming/concept #flashcards/topic/dev-tools What do the status codes `??`, `A `, ` M` and `UU` mean?::`??` a new file Git does not track, `A ` staged, ` M` edited but not staged, `UU` a conflict in a file both sides changed. A blank line means everything is committed.

#flashcards/programming/formula #flashcards/topic/dev-tools Write how Git makes the id of a file and explain each symbol.
?
$$\text{id}=\mathrm{SHA1}\big(\texttt{"blob "}\,\Vert\,\text{size}\,\Vert\,\texttt{NUL}\,\Vert\,\text{content}\big)$$
- $\mathrm{SHA1}$: a hash function that turns any data into a 160-bit fingerprint (40 hex characters). The same input always gives the same fingerprint, and any tiny change gives a completely different one. This is how Git notices changes and checks nothing was corrupted.
- `"blob "` and the size: a header that names the kind of object and its length, so two objects of different kinds do not share an id.
- $\Vert$: joined together. NUL: a zero byte. content: the file's bytes.

#flashcards/programming/example #flashcards/topic/dev-tools A file holds `hello` and a newline (6 bytes). What is the id, and what happens with a capital H?::The hashed bytes are `blob 6`, a zero byte and the 6 content bytes (13 bytes). The id is `ce013625030ba8dba906f756967f9e9ca394464a` (matches `git hash-object`). With `Hello` the id is `e965047ad7c57865823c7d992b1d046ea66edf78`, and 84 of the 160 bits differ.

#flashcards/programming/formula #flashcards/topic/dev-tools Write the chance that two short ids clash and explain each symbol.
?
$$p\ \approx\ 1-\exp\!\left(-\frac{n(n-1)}{2^{\,b+1}}\right)$$
- $n$: the number of objects. The number of pairs $n(n-1)/2$ grows like $n^2/2$, so clashes show up much sooner than expected (the birthday effect).
- $b$: the number of bits shown. Each extra bit halves the chance of a clash. 7 hex letters is $7\times4=28$ bits.
- Each pair clashes with chance $2^{-b}$, so the expected number of clashes is $n(n-1)/2^{b+1}$.

#flashcards/programming/example #flashcards/topic/dev-tools $n=10{,}000$ objects and 7 letters ($b=28$). What is $p$?::$n(n-1)=99{,}990{,}000$, $2^{29}=536{,}870{,}912$, exponent $0.1862$. $p=1-e^{-0.1862}=0.170$. The note's simulation of 2,000 trials gave 0.1715. For the full 160 bits and a billion objects the formula gives about $3.4\times10^{-31}$. About one clash in six with 7 letters.

#flashcards/programming/formula #flashcards/topic/dev-tools Write the size of a diff in changed lines and explain each symbol.
?
$$\#\text{changed lines}=\#\text{removed}+\#\text{added}=(n-\mathrm{LCS})+(m-\mathrm{LCS})=n+m-2\,\mathrm{LCS}$$
- $n$, $m$: the number of lines in the old and new file. They set the most that could change.
- $\mathrm{LCS}$: the length of the longest run of lines that appear in both files in the same order (not necessarily adjacent). These are the lines Git can leave alone. The bigger it is, the smaller the diff.
- $n-\mathrm{LCS}$: old lines to remove. $m-\mathrm{LCS}$: new lines to add.

#flashcards/programming/concept #flashcards/topic/dev-tools Old `a b c d e f g h`, new `a b X d e Y g h`. How many changed lines?::Common lines in order `a b d e g h`, so $\mathrm{LCS}=6$, $n=m=8$. Changed $8+8-2\cdot6=4$: 2 removed (`c`, `f`) and 2 added (`X`, `Y`). Two edited lines show up as 4 changed lines in a diff.

#flashcards/programming/image #flashcards/topic/dev-tools ![[git-operations.svg]]
What does this picture show?
?
How common Git commands move changes between the places. Working files: you edit here. `add` moves changes into the stage. `commit` saves the stage into a branch. `checkout` and `merge` bring a branch's files back into the working files. `push` sends branches to the remote, `fetch` brings the remote's changes into your clone, and `pull` does a fetch and merge in one go. (From the note's caption: Daniel Kinzler, Wikimedia Commons.)

#flashcards/programming/cloze #flashcards/topic/dev-tools Same file, message, author and date in two repositories give the ==same commit id==.
