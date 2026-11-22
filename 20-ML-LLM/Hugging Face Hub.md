---
tags: [LLM, Hugging Face, hosting, models, datasets, versioning, local]
status: growing
level: 4
tier: 2
---

# Hugging Face Hub

The **Hugging Face Hub** is a website where people share AI models, datasets and small demo apps. It works like a library with a public shelf: every model sits in its own folder with a version history, and you can fetch the files from a web address or with a few lines of Python.

## 🟢 1. Simple (age 12)
Think of a huge public library for AI "brains". Each brain has its own shelf with a label that says who made it, what it is good at, and what you may do with it. You can borrow the whole shelf, and the library keeps every older edition too, so you can ask for exactly the one you used last week.

**Example:** you want a small chat model. You go to its shelf on the Hub, read its label, and download its files. A week later the maker updates it. Your copy does not change, and you can still ask for the old edition by its edition number.

## 🔵 2. How it works
1. **Everything lives in a repository** (a "repo"): a folder of files with a history. The Hub's docs say these are Git-based repositories, so they have versions, commits, branches and diffs.
2. **There are three kinds of repo:** models, datasets and Spaces (small apps that run in the browser).
3. **A model repo has a label page**, the *model card*, that tells you what the model is for, its limits and its licence.
4. **Each file has a web address** made of the owner, the repo name, a version and the file name. A version can be a branch name such as `main`, or a fixed commit number.
5. **`main` moves, a commit number does not.** If you use `main`, you get the newest files, and the files may change under you. A commit number always gives the same files.
6. **You fetch files in one of three ways:** click download on the website, call the `huggingface_hub` Python package, or let [[Hugging Face Transformers]] do it when you give `from_pretrained` a model name.
7. **The download goes into a local copy**, so the same file is not fetched twice. With `local_dir` it goes into a plain folder instead (the way I did it).
8. **Some models are "gated":** you must accept the maker's terms before you may download them.

I did not add a picture. The only Hub-related file on Wikimedia Commons that I found was the library logo, which is already in [[Hugging Face Transformers]].

## 🟠 3. Technical detail
- **Asking about a repo costs no download.** One web request gives the repo's metadata: its files with their sizes, its tags, its licence and its latest commit.
- **Web address pattern** (printed by `hf_hub_url` in my run): `https://huggingface.co/<owner>/<repo>/resolve/<revision>/<file>`.
- **The two ways to name a version.** `main` is a branch that moves. A commit hash is fixed. For a repeatable result, pin the full commit hash (printed in the run below), and save it with your notes.
- **Download helpers** (names from the docs I read): `hf_hub_download` for one file and `snapshot_download` for a whole repo. My project used `snapshot_download` with `allow_patterns` to skip files it did not need ([[Local CPU Chatbot]]).
- **The default local cache** (docs summary): `~/.cache/huggingface/hub`, which `HF_HOME` or `HF_HUB_CACHE` can move. Each repo has a folder named like `models--owner--repo` with `blobs` (the files, named by hash), `refs` (which commit a branch points to) and `snapshots` (one folder per commit, holding links to the blobs). On Windows, links may not work, and then the helper keeps plain files in `snapshots` instead, with a less efficient cache (docs summary). I did not open the cache on this PC, though the folders `hub`, `datasets` and `xet` exist under `~/.cache/huggingface`.
- **`hub_peek.py`** (needs `huggingface_hub` and internet; it downloads no file)

```python
"""Ask the Hugging Face Hub about a model repo without downloading any file. Needs: huggingface_hub and an internet connection."""
import os
from huggingface_hub import HfApi, hf_hub_url

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
api = HfApi()

info = api.model_info(REPO, files_metadata=True)                 # one web request: the repo's page as data
print("repo:", info.id, "| pipeline tag:", info.pipeline_tag, "| library:", info.library_name, "| gated:", info.gated, "| private:", info.private)
print("licence tag:", [t for t in info.tags if t.startswith("license")], "| commit:", info.sha[:10])
print("files:", len(info.siblings), "| total bytes:", sum(s.size or 0 for s in info.siblings))
for s in sorted(info.siblings, key=lambda s: -(s.size or 0))[:4]:
    print("  ", s.rfilename, s.size)
print("a file's web address:", hf_hub_url(REPO, "config.json"))
print("same repo at a fixed commit:", hf_hub_url(REPO, "config.json", revision=info.sha))

local = set(os.listdir("qwen2.5-0.5b-instruct"))                 # what my snapshot_download call (with allow_patterns) kept
print("on the Hub but not in my folder:", [(s.rfilename, s.size) for s in info.siblings if s.rfilename not in local])
```
What it printed (huggingface_hub 0.26.2, Python 3.9.6, 18 November 2026):
```
repo: Qwen/Qwen2.5-0.5B-Instruct | pipeline tag: text-generation | library: transformers | gated: False | private: False
licence tag: ['license:apache-2.0'] | commit: 7ae557604a
files: 10 | total bytes: 999604126
   model.safetensors 988097824
   tokenizer.json 7031645
   vocab.json 2776833
   merges.txt 1671839
a file's web address: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json
same repo at a fixed commit: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/config.json
on the Hub but not in my folder: [('.gitattributes', 1519), ('README.md', 4917)]
```
What to look for: the repo has 10 files and 999,604,126 bytes in all, and the weights file (`model.safetensors`, 988,097,824 bytes, the file that [[Safetensors]] opens) is 98.8% of that (my arithmetic: 988,097,824 / 999,604,126 = 0.9885). The file size matches the one in the project note. The two files I left out are `.gitattributes` and `README.md` (the model card), 6,436 bytes together. The licence tag says Apache 2.0, the repo is neither gated nor private, and the commit at the time of my run began with `7ae557604a`. I did not open either web address, so I do not know that the fixed-commit address works as printed, only that the library built it. A different commit may be the latest when you run this.

## 🔴 4. Deep / formal
- **A pinned commit is the only fixed version.** A branch name moves when the owner pushes. Two people who both asked for `main` on different days may get different weights. The commit hash printed above would give the same files each time (the docs say commits are immutable; I did not test that).
- **Reading the licence matters.** The tag said `apache-2.0` for this model. Other models carry other licences or custom terms, some forbid certain uses, and gated models need you to accept terms. The tag is a label, and you should read the model card and licence file before you use a model in anything that matters (my advice; I did not read the Qwen licence text for this note).
- **Trust and safety.** Anyone can upload, so a repo is only as trustworthy as its owner. The docs list user access tokens, access control for organizations, commit signing and malware scanning under security, and I did not test any of them. Prefer [[Safetensors]] files over pickle files, because a pickle can run code on load. Never paste an access token into code that you share.
- **Size on the Hub.** The docs page I read says the Hub hosts "over 3M models, 1M datasets, and 2M AI apps" (read on 18 November 2026; figures change, and I did not count them). Large files are stored with a system called Xet, which splits files into chunks (docs summary).
- **Offline use.** After a download, the files can be used with no internet. My project loads the folder directly, with no Hub call at run time.
- **Limits of this note:** I did not upload a repo, make a token, use a gated model, open the local cache, use Spaces or datasets, or call the Inference Providers or any other paid service.

**What came before, and how this improved on it:**
- **Before:** researchers shared trained weights through links on personal pages, project repositories or cloud drives, each in its own layout (from my memory, I did not check).
- **Its limit:** links die, there was no standard version number, big files do not fit well in ordinary Git, and each project had its own way to load the files.
- **What changed:** one site with Git-style repos for models, datasets and apps, a model card in each, and one download interface. The docs say it offers versioning, commit history, diffs and branches, and that large files are handled by Xet.
- **How much better:** in my run, one request listed all 10 files of a model repo with their sizes and the exact commit (no file was downloaded), and the address of any file at that commit could be built from the repo name. I did not measure a difference in time or effort against an older way, and I did not check when the Hub started.

## Related
[[Hugging Face Transformers]] · [[Safetensors]] · [[GGUF]] · [[Ollama]] · [[Chat Template]] · [[Tokenization]] · [[Quantization]] · [[LLM]] · [[Python]] · [[Agent Safety]] · [[Local CPU Chatbot]]

## Sources
- Run on this PC: huggingface_hub 0.26.2, Python 3.9.6, internet access for one read-only API request about `Qwen/Qwen2.5-0.5B-Instruct` (no file downloaded). The figures in the output above are from my run.
- The Hub documentation start page (https://huggingface.co/docs/hub/index), read on 18 November 2026: repo types, Git-based versioning, model cards, security list, Xet, the model, dataset and Spaces counts.
- The `huggingface_hub` guide "Understand caching" (https://huggingface.co/docs/huggingface_hub/guides/manage-cache), read the same day: the default cache path, the layout of blobs, refs and snapshots, the Windows symlink caveat. I read a page summary, not the full text, and I did not open the cache on this PC.
- From my memory, not checked: how weights were shared before the Hub, and that `from_pretrained` accepts a model name and downloads it.
- No image: nothing on-topic on Wikimedia Commons besides the library logo used in [[Hugging Face Transformers]].
- The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Hugging Face Hub:::A website that hosts Git-based repos of AI models, datasets and small demo apps (Spaces), with versions, model cards and one download interface.

#flashcards/ml/concept #flashcards/topic/llm-serving What are the three kinds of repo on the Hub?::Models, datasets and Spaces (small apps that run in the browser), according to the Hub docs.

#flashcards/ml/concept #flashcards/topic/llm-serving What is the address pattern for a file on the Hub?::`https://huggingface.co/<owner>/<repo>/resolve/<revision>/<file>`, as printed by `hf_hub_url` in my run.

#flashcards/ml/concept #flashcards/topic/llm-serving Why should you pin a commit hash instead of using `main`?::`main` is a branch that moves when the owner pushes, so you may get different files on different days. A commit hash is a fixed version (the docs say commits are immutable; I did not test it).

#flashcards/ml/concept #flashcards/topic/llm-serving What is a model card?::The label page of a model repo: what the model is for, its limits and biases, and its licence (from the Hub docs). Read it before you use the model.

#flashcards/ml/concept #flashcards/topic/llm-serving What does a "gated" model mean?::You must accept the owner's terms before you may download it. The Qwen model in my run was not gated and not private.

#flashcards/ml/example #flashcards/topic/llm-serving What did one API request show about the Qwen2.5-0.5B-Instruct repo?::10 files and 999,604,126 bytes, licence tag `apache-2.0`, not gated, and the weights file `model.safetensors` of 988,097,824 bytes. No file was downloaded.

#flashcards/ml/concept #flashcards/topic/llm-serving What share of the repo's bytes is the weights file?::988,097,824 / 999,604,126 = 0.9885, so 98.8% (my arithmetic).

#flashcards/ml/concept #flashcards/topic/llm-serving Which two files of the repo did my download leave out, and how big are they?::`.gitattributes` (1,519 bytes) and `README.md`, the model card (4,917 bytes), 6,436 bytes together. My `snapshot_download` call used `allow_patterns` that did not match them.

#flashcards/ml/concept #flashcards/topic/llm-serving How is the `huggingface_hub` local cache laid out?
?
- The default folder is `~/.cache/huggingface/hub` (`HF_HOME` or `HF_HUB_CACHE` can move it).
- Each repo gets a folder like `models--owner--repo`.
- Inside: `blobs` (files named by hash), `refs` (which commit a branch points to) and `snapshots` (one folder per commit, with links to the blobs).
- This is from the docs summary. I did not open the cache on this PC.

#flashcards/ml/cloze #flashcards/topic/llm-serving A model's ==licence== tag, for example apache-2.0, tells you what you may do with it, but you should still read the model card and licence file.

#flashcards/ml/concept #flashcards/topic/llm-serving Why prefer safetensors files from the Hub over pickle files?::Anyone can upload to the Hub, and a pickle file can run code when loaded. A safetensors file holds only data (see the Safetensors note).

#flashcards/ml/concept #flashcards/topic/llm-serving What did I not test about the Hugging Face Hub?::Uploading a repo, access tokens, gated models, the local cache on this PC, Spaces, datasets, malware scanning, the Inference Providers, and whether the printed fixed-commit address opens.
