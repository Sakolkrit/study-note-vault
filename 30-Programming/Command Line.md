---
tags: [command-line, shell, bash, powershell, programming, tooling]
status: growing
level: 4
tier: 1
---

# Command Line

The **command line** is a way of talking to your computer by typing instead of clicking. You type a short instruction (a *command*), press Enter, and the computer does it and prints a reply. The program that reads what you type is called the **shell**. Programmers use it all the time, because a typed command can be saved, repeated, combined with others, and run on a computer far away. Tools such as [[Git]] and [[Python]] are mostly run from here.

## 🟢 1. Simple (age 12)
Imagine a very obedient helper who sits inside the computer. You can only talk to them by writing short notes. "Show me what is in this folder." "Make a new folder called photos." "Copy this file there." The helper does exactly what the note says and writes back. It is slower to learn than clicking, but you can write many notes at once, and you can hand the same list of notes to a friend and they will get the same result.

**Example:** You have 500 photos named `img1.jpg` to `img500.jpg` and you want to move them all into a folder. Clicking takes ages. One typed command, `mv *.jpg photos/`, moves them all at once, because the `*` means "anything".

## 🔵 2. How it works
1. **There is always a "where am I".** The shell is always inside one folder, called the **working directory**. A command like `ls` (list) shows what is in *that* folder.
2. **Paths say where things are.** An *absolute path* starts from the very top of the disk (`/` on Linux and macOS and in Git Bash, or `C:\` on Windows). A *relative path* starts from where you are. `.` means "here" and `..` means "one folder up".
3. **A command has a name, then options and arguments.** In `ls -l notes`, `ls` is the program, `-l` is an option, and `notes` is what to work on. Programs are found by name in a list of folders called the `PATH`.
4. **The shell tidies up your words first.** It expands `*.txt` into the list of matching files, replaces `$NAME` with the value of a variable, and splits what you typed at spaces. Quotes control this: single quotes `'...'` keep everything as plain text, double quotes `"..."` still fill in variables, and both stop a file name with a space in it from being split in two.
5. **Programs talk through three channels.** Each program reads from *standard input* (`stdin`), writes results to *standard output* (`stdout`), and writes error messages to *standard error* (`stderr`). Normally input is your keyboard and the two outputs go to your screen.
6. **You can connect programs.** A **pipe** `|` sends one program's output straight into the next program's input. A **redirect** sends output to a file (`>` replaces the file, `>>` adds to it) or reads input from a file (`<`).
7. **Every program says whether it worked.** When it finishes it returns a number, the *exit code*: 0 means success, anything else means some kind of failure.

![[unix-pipeline.svg]]
*A pipeline of three programs in a text terminal. The keyboard feeds Program 1 through `stdin`. Each program's `stdout` becomes the next program's `stdin`, and the last one's `stdout` goes to the display. All three send error messages (`stderr`) straight to the display instead of down the pipe. Image: XcepticZP (new layout) after TuukkaH (original diagram), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Pipeline.svg), public domain.*

## 🟠 3. Technical detail
- **Which shell?** The word "command line" covers several shells. **Bash** (also in Git for Windows, as "Git Bash") is the standard on Linux and macOS. **PowerShell** is the modern Windows shell. The older `cmd.exe` also exists on Windows, and I did not test it. This note's code ran **both Bash and PowerShell** on the same Windows computer.
- **Bash against PowerShell:** the main difference is that Bash programs pass **text** to each other, and PowerShell passes **objects** (things with named properties, such as a file's `Length`). The ideas are the same: working directory, paths, pipes, exit codes, variables. The code runs each idea in both where it can.

| Idea | Bash (run) | PowerShell (run) |
|---|---|---|
| list files matching `*.txt` | `ls *.txt` | `Get-ChildItem *.txt` |
| count them | `ls *.txt \| wc -l` | `(Get-ChildItem *.txt).Count` |
| a variable | `X=hi; echo "$X"` | `$x = "hi"; "$x"` |
| an environment variable | `export FOO=bar` | `$env:FOO = "bar"` |
| exit code of the last program | `$?` | `$LASTEXITCODE` |
| text with no variables filled in | `echo '$HOME'` | `'$HOME'` |

- **Common commands** (Bash names, with the PowerShell word in brackets): `pwd` (`Get-Location`) shows where you are, `cd` changes folder, `ls` (`Get-ChildItem`) lists, `mkdir` makes a folder, `cp`, `mv` and `rm` copy, move and delete, `cat` shows a file, `grep` finds lines containing text, `sort` and `uniq -c` order and count lines, `wc -l` counts lines, `head` shows the first lines. PowerShell also accepts many of the Bash names as aliases (from memory; I did not test which).
- **`rm` is permanent:** there is no recycle bin. Check what a pattern like `*.txt` matches by running `ls` with the same pattern first. The code only creates and reads files inside a temporary folder.
- **Pipeline example:** `tr -s ' ' '\n' < words.txt | sort | uniq -c | sort -rn | head -3` splits a text into words, sorts them, counts repeats, sorts by the count, and keeps the top 3. The code checks it against Python's `collections.Counter` on a short sentence, and both give the, and, cat with counts 5, 3 and 2. When several words tie, the shell and Python may break the tie differently, so I chose a sentence with no ties among the top 3.
- **Redirects:** `>` replaces and `>>` appends (the code shows 2 lines after one of each, then 1 line after `>` again). `2>` sends only error messages to a file, so the screen stays quiet and the file is not empty.
- **Exit codes and chaining:** `a && b` runs `b` only if `a` succeeded (exit code 0), and `a || b` runs `b` only if `a` failed. `grep -q` finds a word silently and reports the result only through its exit code: 0 for found, 1 for not found. A command that does not exist gives 127 in Bash (shown in the code).
- **Environment variables:** a plain variable stays inside the shell. A child program sees it only if you `export` it (the code shows an empty `[]` first and `[bar]` after `export`). `PATH` is a list of folders that the shell searches for commands.
- **Quoting traps:** a file named `my file.txt` unquoted is two words to the shell, and quoted is one (the code counts 2 against 1). Always quote variables that hold paths: `"$f"`.
- **Scripts:** a list of commands saved in a file is a *shell script*. I did not write or run a script file here, only one-line commands.
- **Limits:** the code ran Git Bash and Windows PowerShell 5.1 (the version that is installed here) in a temporary folder, and the output may differ on Linux, macOS or newer PowerShell. It tests basic ideas only, not scripts, loops, `if` statements or functions, and it does not time anything.

The code covers seven things: (1) folders and paths, (2) wildcards and quoting, (3) a pipeline checked against Python, (4) redirection, (5) exit codes and chaining, (6) environment variables and `PATH`, (7) the same ideas in PowerShell.

```python
import subprocess, tempfile, os, collections
D = tempfile.mkdtemp(prefix='cldemo')
def sh(cmd, shell='bash'):
    args = ['bash', '-c', cmd] if shell == 'bash' else ['powershell', '-NoProfile', '-Command', cmd]
    r = subprocess.run(args, cwd=D, capture_output=True, text=True)
    return r.stdout.strip().replace('\r', '')
# 1) where am I? paths are relative to the working directory; . is here, .. is the parent
print('walk:', sh('mkdir -p proj/data && cd proj/data && basename "$PWD" && cd .. && basename "$PWD" && ls').replace('\n', ' > '))
print('absolute path of the root folder:', sh('cd / && pwd'), '| relative path from proj/data to proj:', '..')
# 2) globs and quoting: the shell expands * before the program runs; quotes stop the splitting
sh('touch a.txt b.txt c.md "my file.txt"')
print('ls *.txt finds', sh('ls *.txt | wc -l'), 'files | ls *.md finds', sh('ls *.md | wc -l'))
print('single quotes keep $HOME as text:', sh("echo '$HOME'"), '| double quotes expand a variable:', sh('X=hi; echo "$X there"'))
print('unquoted "my file.txt" is', sh('f="my file.txt"; set -- $f; echo $#'), 'words | quoted it is', sh('f="my file.txt"; set -- "$f"; echo $#'), 'word')
# 3) pipes: the output of one program is the input of the next
words = 'the cat and the dog and the cat and the bird the'
open(os.path.join(D, 'words.txt'), 'w', newline='\n').write(words + '\n')
pipe = sh("tr -s ' ' '\\n' < words.txt | sort | uniq -c | sort -rn | head -3")
print('pipeline top 3 words:', [tuple(l.split()) for l in pipe.split('\n')], '| Python Counter:', [(str(c), w) for w, c in collections.Counter(words.split()).most_common(3)])
# 4) redirection: > replaces a file, >> adds to it, 2> catches error messages, < feeds a file in
print('lines after > then >>:', sh('echo one > f.txt; echo two >> f.txt; wc -l < f.txt'), '| after > again:', sh('echo three > f.txt; wc -l < f.txt'))
print('error message went to err.txt, not the screen:', repr(sh('ls nothing-here 2> err.txt')), '| err.txt is not empty:', sh('test -s err.txt && echo yes'))
# 5) exit codes: 0 means success; && runs the next command only on success, || only on failure
print('true ->', sh('true; echo $?'), '| false ->', sh('false; echo $?'), '| grep finds ->', sh('grep -q cat words.txt; echo $?'), '| grep misses ->', sh('grep -q zebra words.txt; echo $?'))
print('chains:', sh('true && echo yes-after-success'), '|', sh('false || echo fallback-after-failure'), '|', repr(sh('false && echo never-printed')))
# 6) environment variables are copied to child programs only if exported; PATH is the list of folders searched for commands
print('plain variable seen by child:', sh("FOO=bar; sh -c 'echo \"[$FOO]\"'"), '| exported:', sh("export FOO=bar; sh -c 'echo \"[$FOO]\"'"))
print('command found on PATH:', sh('command -v ls >/dev/null && echo found'), '| with an empty PATH the same name fails with code', sh('PATH=/nonexistent; nosuchcmd 2>/dev/null; echo $?'))
# 7) the same ideas in PowerShell (it passes objects, not text)
print('*.txt count, bash:', sh('ls *.txt | wc -l'), '| PowerShell:', sh('(Get-ChildItem *.txt).Count', 'ps'), '| files longer than 5 bytes, bash:', sh('find . -maxdepth 1 -type f -size +5c | wc -l'), '| PowerShell:', sh('(Get-ChildItem -File | Where-Object Length -gt 5 | Measure-Object).Count', 'ps'))
print("PowerShell single quotes literal:", sh("'$HOME'", 'ps'), '| double quotes expand:', sh('$x = "hi"; "$x there"', 'ps'), '| env var:', sh('$env:FOO = "bar"; "[$env:FOO]"', 'ps'))
print('PowerShell exit code of a failing program:', sh('cmd /c exit 3; $LASTEXITCODE', 'ps'), '| of a succeeding one:', sh('cmd /c exit 0; $LASTEXITCODE', 'ps'), '| pipeline words:', sh("('the cat the dog' -split ' ' | Group-Object | Sort-Object Count -Descending | Select-Object -First 1).Name", 'ps'))
# walk: data > proj > data
# absolute path of the root folder: / | relative path from proj/data to proj: ..
# ls *.txt finds 3 files | ls *.md finds 1
# single quotes keep $HOME as text: $HOME | double quotes expand a variable: hi there
# unquoted "my file.txt" is 2 words | quoted it is 1 word
# pipeline top 3 words: [('5', 'the'), ('3', 'and'), ('2', 'cat')] | Python Counter: [('5', 'the'), ('3', 'and'), ('2', 'cat')]
# lines after > then >>: 2 | after > again: 1
# error message went to err.txt, not the screen: '' | err.txt is not empty: yes
# true -> 0 | false -> 1 | grep finds -> 0 | grep misses -> 1
# chains: yes-after-success | fallback-after-failure | ''
# plain variable seen by child: [] | exported: [bar]
# command found on PATH: found | with an empty PATH the same name fails with code 127
# *.txt count, bash: 6 | PowerShell: 6 | files longer than 5 bytes, bash: 3 | PowerShell: 3
# PowerShell single quotes literal: $HOME | double quotes expand: hi there | env var: [bar]
# PowerShell exit code of a failing program: 3 | of a succeeding one: 0 | pipeline words: the
```
Reading the output (Git Bash and Windows PowerShell 5.1, Python 3.9):
- **Paths:** the walk went into `proj/data`, up to `proj`, and listed its contents (`data`). The absolute path of the top folder is `/` in Git Bash.
- **Wildcards:** `ls *.txt` found 3 files (including the one with a space in its name) and `ls *.md` found 1.
- **Quotes:** `'$HOME'` stayed as the text `$HOME`, `"$X there"` became `hi there`, and the unquoted file name split into 2 words against 1 when quoted.
- **Pipeline:** the top 3 words were `the` (5), `and` (3) and `cat` (2), exactly the same as Python's `Counter`.
- **Redirects:** 2 lines after `>` and `>>`, then 1 line after `>` again. The error message went to `err.txt` (the screen showed nothing), and the file was not empty.
- **Exit codes:** `true` gave 0, `false` gave 1, `grep` gave 0 when it found `cat` and 1 for `zebra`. `&&` ran after success, `||` ran after failure, and `false && …` printed nothing.
- **Environment:** the child saw `[]` for a plain variable and `[bar]` after `export`. With `PATH` pointing at a folder that does not exist, an unknown command gave exit code 127.
- **PowerShell:** the `*.txt` count was 6 in both shells at that moment (the folder had gained more files by then), and 3 files were longer than 5 bytes in both. Single quotes kept `$HOME` as text, double quotes filled in the variable, `$env:FOO` showed `[bar]`, `$LASTEXITCODE` gave 3 and 0 for programs that exit with 3 and 0, and a PowerShell pipeline found `the` as the most common word of `the cat the dog`.
- **Limits:** one-line commands in a temporary folder, nothing timed.

## 🔴 4. Deep / formal
This topic has no formulas, so it has no equations to work through. The precise ideas are the rules the shell follows and the contract between programs, and each rule below is one I ran in the code.

- **The order the shell works in** (from memory; the code checks several pieces, not the whole order): it reads the line, splits it into words (respecting quotes), expands variables and wildcards, then starts the program with the resulting list of words. The program never sees the `*`, only the file names it was turned into. That is why `ls *.txt` lists files the same way however `ls` is written, and why a pattern that matches nothing may reach the program unexpanded (I did not test that case).
- **Three channels, numbered:** standard input, standard output and standard error are, from memory, numbered 0, 1 and 2, which is why you write `2>` for errors. The figure shows the three roles, and the code uses `2>`, but I did not test the numbers themselves in any other way.
- **A pipe is a connection, not a file:** in `a | b`, `b` can start reading while `a` is still writing, and nothing is stored on disk (from memory; I did not test it). Only `stdout` goes down the pipe, so error messages from `a` still reach the screen, as the figure shows.
- **Exit code contract:** 0 is success and any other number is failure, and the meaning of non-zero numbers belongs to each program. The code saw 1 from `grep` for "not found" and `false`, and 127 from Bash for "command not found". Whether `grep` also returns 2 on a real error is from memory, and the code did not test it. `&&` and `||` look only at whether the code is 0.
- **Child programs get a copy of the environment, not a link:** exporting a variable lets child programs read it, but a child can never change the parent shell's variables. The code shows the first half (visible only after `export`). I did not test the second half.
- **Two shells, two models:** Bash joins programs by text, so every program must agree on how to split text into columns. PowerShell joins them by objects, so `Where-Object Length -gt 5` can ask for a property by name. The code gave the same answer (3 files) both ways. The price is that PowerShell's output is richer but tied to its own commands, from memory.
- **Limits of text pipelines:** a filename that contains a newline can break a pipeline that treats one line as one file. I did not test this and do not cover how to handle it.

**Other points (from memory, not checked here):**
- The pipe idea came from Unix in the early 1970s and is usually credited to Douglas McIlroy. I did not check the date or the credit.
- Bash stands for "Bourne-again shell". The Windows Terminal is an app that can host several shells. I did not check either.
- On macOS the default shell has been zsh in recent versions. I did not check this.
- Windows can also run Linux shells through the Windows Subsystem for Linux (WSL). I did not use WSL here.

**What came before, and how this improved on it:**
- **Before:** do every task by clicking in a window, one item at a time.
- **Its limit:** a job on 1,000 files means 1,000 clicks, and nothing is saved for next time.
- **What changed:** type a command, which can take arguments, be saved in a script and run again. Unix shells date from the 1970s (Thompson shell, 1971; from memory, not checked).
- **How much better:** in the code below, one program run from the command line with the folder as its argument renamed 1,000 of 1,000 files with exit code 0, instead of 1,000 manual renames. The snippet starts the program from Python, so it shows the idea rather than typing in a terminal.

**Check it yourself:**

```python
import os, subprocess, sys, tempfile
d = tempfile.mkdtemp()
for i in range(1000): open(os.path.join(d, f"photo{i}.jpeg"), "w").close()
prog = "import os,sys;[os.rename(os.path.join(sys.argv[1],f),os.path.join(sys.argv[1],f.replace('.jpeg','.jpg'))) for f in os.listdir(sys.argv[1])]"
r = subprocess.run([sys.executable, "-c", prog, d])
names = os.listdir(d)
print("one command renamed", sum(n.endswith(".jpg") for n in names), "of", len(names), "files; exit code", r.returncode)
print("by hand in a file window: 1,000 rename actions; the command can also be saved and run again")
```

## Related
[[Git]] · [[Python]] · [[Virtual Environments]] · [[NumPy]] · [[Pandas]] · [[PyTorch]] · [[TensorFlow]] · [[Hash Tables]] · [[Sorting]] · [[Regular Expressions]]
- [[pytest]] · [[Docker]]

## Sources
- GNU Bash reference manual (gnu.org) and Microsoft's PowerShell documentation (learn.microsoft.com), on quoting, redirection, pipes and exit codes (from memory; I did not open them for this note).
- Figure: [Pipeline.svg](https://commons.wikimedia.org/wiki/File:Pipeline.svg) (XcepticZP, after TuukkaH, public domain), on Wikimedia Commons. I viewed it before writing the caption.
- The code and every printed value were run for this note (Git Bash and Windows PowerShell 5.1, Python 3.9, in a temporary folder that nothing outside touches). The code was not timed.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/dev-tools Command line:::Talking to the computer by writing short typed commands. Slower to learn than clicking, but you can write many at once and save them.

#flashcards/programming/reverse #flashcards/topic/dev-tools Working directory:::The folder the shell is always "inside". A command like `ls` lists that folder.

#flashcards/programming/concept #flashcards/topic/dev-tools Absolute path against relative path?::An absolute path starts from the top of the disk (`/` on Linux, macOS and Git Bash, or `C:\` on Windows). A relative path starts from where you are. `.` means here and `..` means one folder up.

#flashcards/programming/concept #flashcards/topic/dev-tools In `ls -l notes`, what are `ls`, `-l` and `notes`?::`ls` is the program, `-l` is an option, and `notes` is the argument (what to work on). Programs are found by name in the folders listed in `PATH`.

#flashcards/programming/concept #flashcards/topic/dev-tools What do single quotes and double quotes do in the shell?::Single quotes `'...'` keep everything as plain text. Double quotes `"..."` still fill in variables. Both stop a file name with a space from being split in two.

#flashcards/programming/concept #flashcards/topic/dev-tools What are the three channels a program talks through?::Standard input (`stdin`), standard output (`stdout`) and standard error (`stderr`). Normally input is the keyboard and the two outputs go to the screen.

#flashcards/programming/concept #flashcards/topic/dev-tools What does a pipe `|` do?::It sends one program's output straight into the next program's input.

#flashcards/programming/example #flashcards/topic/dev-tools Redirect `>` against `>>`, and what does `2>` do?::`>` replaces the file, `>>` adds to it. `2>` sends only error messages to a file, so the screen stays quiet.

#flashcards/programming/concept #flashcards/topic/dev-tools `a && b` against `a || b`?::`a && b` runs `b` only if `a` succeeded (exit code 0). `a || b` runs `b` only if `a` failed.

#flashcards/programming/concept #flashcards/topic/dev-tools What does `grep -q` report?::Nothing on screen. It reports only through its exit code: 0 for found, 1 for not found.

#flashcards/programming/concept #flashcards/topic/dev-tools Does a child program see a plain shell variable?::No. A child sees it only after `export`. The note's code showed `[]` before `export` and `[bar]` after.

#flashcards/programming/concept #flashcards/topic/dev-tools A file named `my file.txt` is used unquoted. What goes wrong?::The shell splits it into two words. Quoted, it is one. Always quote variables that hold paths: `"$f"`.

#flashcards/programming/reverse #flashcards/topic/dev-tools Bash and PowerShell differ mainly in what programs pass to each other:::Bash passes text, PowerShell passes objects (things with named properties such as a file's `Length`).

#flashcards/programming/concept #flashcards/topic/dev-tools Set an environment variable in Bash and in PowerShell?::Bash: `export FOO=bar`. PowerShell: `$env:FOO = "bar"`.

#flashcards/programming/concept #flashcards/topic/dev-tools Exit code of the last program in Bash and in PowerShell?::Bash: `$?`. PowerShell: `$LASTEXITCODE`.

#flashcards/programming/concept #flashcards/topic/dev-tools List files matching `*.txt` in Bash and in PowerShell?::Bash: `ls *.txt`. PowerShell: `Get-ChildItem *.txt`.

#flashcards/programming/cloze #flashcards/topic/dev-tools `rm` has no ==recycle bin==, so check what a pattern like `*.txt` matches by running `ls` with the same pattern first.

#flashcards/programming/image #flashcards/topic/dev-tools ![[unix-pipeline.svg]]
What does this picture show?
?
A pipeline of three programs in a text terminal. The keyboard feeds Program 1 through `stdin`. Each program's `stdout` becomes the next program's `stdin`, and the last one's `stdout` goes to the display. All three send error messages (`stderr`) straight to the display instead of down the pipe. (From the note's caption: XcepticZP after TuukkaH, Wikimedia Commons, public domain.)

#flashcards/programming/example #flashcards/topic/dev-tools What does `tr -s ' ' '\n' < words.txt | sort | uniq -c | sort -rn | head -3` do?::It splits a text into words, sorts them, counts repeats, sorts by the count, and keeps the top 3. In the note's code the top 3 were `the` (5), `and` (3) and `cat` (2), matching Python's `Counter`.
