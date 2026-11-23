# Vault Guide

## Folders
| Folder | Holds |
|---|---|
| 00-Inbox | new notes land here, sort later |
| 10-Math | linear algebra, calculus, probability, optimization |
| 20-ML-LLM | transformers, RAG, training, agents, papers |
| 30-Programming | python, libraries (NumPy, pandas, PyTorch, TensorFlow), tooling, snippets |
| 40-Algorithms | data structures, algorithms, Big-O, with runnable code |
| 50-Courses | courses: an index note and one summary note per lecture, linking to concept notes (first course: CME 295) |
| 55-Books | books: an index note and one summary note per chapter, linking to concept notes (summaries only of what was read) |
| 60-Projects | one note per project: goal, result, exact steps that were run, what went wrong (first: Iris Classifier Service) |
| 70-Research | one note per research paper (AlexNet, YOLO, Attention Is All You Need...) |
| 90-Templates | Concept, Lecture, Course, Book, Chapter, Project, Paper |
| 99-Attachments | images, PDFs |

## Rules
1. **One concept = one note.** Name = the term (`RAG`, `Embedding`).
2. **Progressive depth.** Concept notes go 🟢 Simple (age 12) → 🔵 How it works → 🟠 Technical → 🔴 Deep/formal (the colour dot in each heading shows the level). Stop at the level you understand; set `level:` in frontmatter and deepen later.
3. **Every hard word is a `[[link]]`.** Unwritten link = to-do (shows in graph).
4. **Tags = topic families** (`RAG`, `LLM`, `transformer`, `math`). Add every parent field the topic belongs to.
5. **Images:** real ones found online, saved to `99-Attachments`, with source URL under the image. No AI-generated images.
6. **Math** in `$...$` / `$$...$$`. **Code** in fenced blocks.
7. **Every display equation gets a `Where:` list** right under it: each symbol, what it is in plain words, and **Matters:** what it does to the result.
8. **Every `Where:` list is followed by an `Example (tiny numbers):`**: small, easy numbers worked step by step, ending with what the answer means.
9. **History block:** near the end of each concept note, **What came before, and how this improved on it** says what the older method was, its limit, what changed, and how much better (a number measured in the note's code when it can be run).

## Use
New note → `Ctrl+P` → *Templates: Insert template* → Concept / Lecture / Course / Book / Chapter / Project / Paper.

Example: [[RAG]]

## Setup: PC and Android (sync through a private GitHub repo)

Goal: edit on the PC, read and edit on the phone, with the laptop switched off. GitHub is the middle. Steps written from how these tools work; I ran the PC side (repo created and pushed), **I did not run the Android side**, so treat it as a guide and expect small differences.

### 0. Once, on any computer
1. Install [Obsidian](https://obsidian.md) and [Git](https://git-scm.com/downloads).
2. Make a free GitHub account.
3. Create a **private** repo (empty, no README). Private matters: notes are personal.
4. Make a **personal access token**: GitHub → Settings → Developer settings → Personal access tokens → *Tokens (classic)* → scope `repo`. Copy it once and keep it in a password manager. It works like a password. Never put it in a note or the repo.

### 1. PC (Windows / Mac / Linux)
**Option A: share this vault.** In the vault folder:
```bash
git init -b main
git add -A
git commit -m "first commit"
git remote add origin https://github.com/YOUR-NAME/YOUR-REPO.git
git push -u origin main
```
(If the GitHub CLI `gh` is installed: `gh repo create YOUR-REPO --private --source . --push` does the same in one line.) Windows asks you to sign in on the first push; use your username and the token as the password, or the browser sign-in.

**Option B: you got this vault from someone else.** Clone it, then open the folder in Obsidian (*Open folder as vault*):
```bash
git clone https://github.com/THEIR-NAME/THEIR-REPO.git
```
Private repo? The owner must add you as a collaborator first. To keep your own copy separate, fork it or push it to a new repo of your own.

**Then in Obsidian:** Settings → Community plugins → turn off Restricted mode → Browse → install **Spaced Repetition** (flashcards; the notes use its card syntax) and **Obsidian Git** (sync). In Obsidian Git settings you can set auto commit-and-sync every N minutes, or run *Obsidian Git: Commit-and-sync* from `Ctrl+P`.

Put a `.gitignore` in the vault root so device-only files do not clash between PC and phone. This vault uses:
```text
.obsidian/workspace.json
.obsidian/workspaces.json
.obsidian/cache
.claude/
.trash/
```
Add folders you do not want online (big downloads, other people's repos) the same way.

### 2. Android
1. Install **Obsidian** from Google Play.
2. Create a folder for the vault, then in Obsidian: *Open folder as vault* (or *Create new vault* in that folder). Turn off Restricted mode for community plugins and install **Obsidian Git** (and **Spaced Repetition**).
3. Get the notes onto the phone. Make sure the **Git** (Obsidian Git) plugin is switched **on**, not just installed. Open the command palette (swipe down on a note, or the ⋮ menu), type `clone`, and run **Obsidian Git: Clone an existing remote repo**. You are asked these things in turn. This order was reported by the vault owner from a real phone run (I did not see the screens myself); the wording may differ in other plugin versions:
   1. **URL:** paste the HTTPS address `https://github.com/YOUR-NAME/YOUR-REPO.git`.
   2. **Which directory to put it in:** choose the **root** of the vault (the vault folder itself). Do not type `.` or a folder name that does not exist yet; that gave a "directory does not exist" error.
   3. **Does the phone already have a `.obsidian` config in this vault?** Answer **No**, so the repo's own settings are used.
   4. **An authentication error appears.** This is expected the first time, because a private repo needs a login. Carry on.
   5. **Username:** your GitHub username.
   6. **Password:** paste the **token** from step 0, not your GitHub password.
   7. It now downloads the repo into the vault. Wait; a few hundred files can take several minutes and may look frozen. Then restart Obsidian if the notes do not show.
   - Do not type `git clone ...` anywhere. Obsidian is not a terminal; the command palette runs the plugin's clone for you.
   - If you answered a prompt wrongly or it stops half way, delete the half-filled vault, create a fresh empty one, and run the clone again.
4. Daily use: **pull before you start**, **commit and push when you finish** (*Obsidian Git: Pull*, then *Commit-and-sync*). Auto sync on a timer works but is less reliable on mobile.

Known limits of Obsidian Git on Android (from the plugin's own mobile notes as I remember them, not tested here):
- Only the HTTPS address with a token works. SSH keys do not.
- It uses a pure-JavaScript Git, so a big vault (hundreds of MB, many images) can be slow or run out of memory. This vault is about 44 MB; I did not test it on a phone.
- Merge conflicts are hard to fix on a phone. Avoid editing the **same note** on both devices before syncing.

### 2b. Author name and email (once per device)
Git labels every commit with a name and an email. Any values work; they are only a label. To keep your real email out of the history, use your GitHub **noreply** address: `ID+USERNAME@users.noreply.github.com` (find the ID at github.com → Settings → Emails, or ask `gh api user`).
- **PC:** in the vault folder run `git config user.name "YOUR-NAME"` and `git config user.email "ID+USERNAME@users.noreply.github.com"`.
- **Android:** Obsidian → Settings → Obsidian Git → *Author name for commit* and *Author email for commit* (names may differ by version). Obsidian Git asks for these the first time you pull.

### 2c. Pull and push, step by step
**Pull** = download what changed on GitHub. **Push** = upload what you changed. Pull first, edit, then push. Always in that order.

**PC, with Obsidian Git (easiest):**
1. Open the vault, press `Ctrl+P`.
2. Type `pull`, run **Obsidian Git: Pull**. Wait for the notice "Pulled".
3. Edit notes as usual.
4. When done: `Ctrl+P` → **Obsidian Git: Commit-and-sync** (it commits, pulls, then pushes). Or turn on auto commit-and-sync in the plugin settings.

**PC, in a terminal** (open it in the vault folder):
```bash
git pull
# ...edit notes...
git add -A
git commit -m "describe the change"
git push
```
`git status` shows what changed. If `git pull` says you have local changes, commit them first.

**Android, with Obsidian Git:**
1. Open the vault. Open any note.
2. Open the command palette (swipe down on the note, or the ⋮ menu).
3. Type `pull`, tap **Obsidian Git: Pull**. If asked for the author name and email, fill them in (section 2b), then run Pull again.
4. If asked for a username and password, give your GitHub username and the **token** as the password.
5. Wait for the "Pulled" notice. Large changes can take a while.
6. Edit notes.
7. When done: command palette → type `commit` → **Obsidian Git: Commit-and-sync** (or *Commit all changes*, then *Push*).

**Check it worked:** edit a note on one device, push, pull on the other, and look for the change. On GitHub the repo page shows the latest commit and its time.

**If a pull fails:**
- *Authentication error:* the token is wrong, expired, or lacks the `repo` scope. Make a new one and enter it again.
- *Local changes would be overwritten:* commit your changes first (Commit-and-sync), then pull.
- *Merge conflict:* see the habits below. Fix the note, then commit and sync again.
- *Nothing happens on Android:* make sure the **Git** plugin is switched on, and check you have internet.

### 3. Habits that avoid trouble
- **One device at a time.** Sync (pull, then push) before you switch devices.
- **Flashcard schedules live inside notes** (`<!--SR:...-->` lines). Reviewing on the phone changes those notes; sync, so the PC gets the schedule and no progress is lost.
- If a sync stops with a conflict, open the note, keep the right parts, delete the three Git marker lines (one starts with seven `<`, one is a line of equals signs, one starts with seven `>`), then commit and sync again.
- If Obsidian Git on the phone is too fragile, other ways to share the folder: **Syncthing** (free, phone-to-PC directly, but the PC must be on), or a cloud folder with the *Remotely Save* plugin (OneDrive, Dropbox, S3). Not tested here.

### 4. Sharing the vault with others
- **Read-only for friends:** add them to the private repo as collaborators (Settings → Collaborators), or make the repo public. A public repo shows every note and picture to everyone, so check the pictures' licences first (each caption names its licence).
- **Never commit** tokens, passwords or personal files. Check `git status` before the first push. If a secret gets pushed, change it at once; deleting the file later does not remove it from history.
- Third-party material (for example a copy of a course's code) may not be yours to share. This vault keeps such copies out of Git.

## Indexes
Start here to find any note: [[ML and LLM Index]] · [[Programming Index]] · [[Math Index]] · [[Algorithms Index]] · [[Projects Index]] · [[Research Index]] · [[Books Index]] · courses: [[CME 295]] · [[Machine Learning Specialization]]
