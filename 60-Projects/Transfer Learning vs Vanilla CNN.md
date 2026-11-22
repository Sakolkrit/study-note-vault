---
tags: [project, python, pytorch, torchvision, cnn, transfer-learning, resnet, image-classification, cpu]
status: done
stack: Python 3.9, PyTorch (CPU), torchvision, Pillow
goal: classify ants versus bees from only 244 training photos, and measure how much a ResNet-18 pre-trained on ImageNet beats a plain small CNN trained from scratch, with fewer images and with longer training as extra checks
---

# Transfer Learning vs Vanilla CNN

> **In one line:** take a tiny photo set (244 pictures of ants and bees) and compare four ways to train an image classifier on a laptop CPU: a plain small [[Convolutional Neural Network]], the same kind of big network as the winner ([[ResNet]]) trained from nothing, and the big network after [[Transfer Learning]] with its old knowledge kept frozen or fine-tuned.

## Goal
Done means five checks pass: (1) the data is downloaded and counted, and the pre-trained weights are the published file, (2) four models are trained on the same pictures with the same settings except the ones that must differ, (3) every setting is run with 3 random seeds and I report spread as well as the average, (4) I test two fairness worries I had: "maybe the plain network just needs fewer pictures or more training time", and (5) I say what the numbers can and cannot show.

## Result in short
- **Data:** 244 training photos (123 ants, 121 bees) and 153 validation photos (70 ants, 83 bees). A model that always says "bee" scores **54.2%** (83 of 153). I used 128 by 128 pixel crops so everything would run on a CPU.
- **The four models** (validation accuracy after 15 epochs, mean of 3 seeds, with the spread in brackets):

| Model | Numbers it trains | Val accuracy | Per seed |
|---|---|---|---|
| Vanilla CNN, from scratch | 98,178 | **70.8%** (sd 3.9) | 73.2, 65.4, 73.9 |
| ResNet-18, from scratch | 11,177,538 | **67.3%** (sd 2.7) | 67.3, 64.1, 70.6 |
| ResNet-18, pre-trained, frozen (train only a new 2-class layer) | 1,026 | **88.9%** (sd 1.4) | 88.2, 87.6, 90.8 |
| ResNet-18, pre-trained, fine-tuned (train everything, small steps) | 11,177,538 | **90.6%** (sd 0.8) | 90.8, 89.5, 91.5 |

- **The headline:** starting from ImageNet knowledge is worth about **18 to 23 points** here (18.1 and 19.8 points for frozen and fine-tuned against the vanilla CNN, and 21.6 and 23.3 against the from-scratch ResNet-18; my arithmetic). The frozen model trains only 1,026 numbers and still beat the vanilla CNN. The difference between frozen (88.9%) and fine-tuned (90.6%) is about the size of the noise (see "How to read the numbers").
- **A big network is not the cure by itself:** the same ResNet-18 with random starting weights did **not** beat the small vanilla CNN (67.3% against 70.8%). The gain comes from the pre-training, not from the architecture.
- **Fewer pictures:** the frozen pre-trained model trained on just **61** pictures (25%) scored **86.9%**, which beats the vanilla CNN trained on all **244** (70.8%) by 16 points.
- **More training does not close the gap:** with 60 epochs (4 times longer) the vanilla CNN reached **74.1%** (sd 1.9, 3 seeds) and the from-scratch ResNet-18 scored 61.4% (one seed only). The pre-trained models were at 88.9% and 90.6% after 15 epochs.
- **Speed (rough, from this CPU):** vanilla about 91 s, frozen about 152 s, fine-tuned about 304 s and ResNet-18 from scratch about 312 s for 15 epochs. These times moved by a lot between runs (the same seed-0 ResNet-18 from-scratch run took 387 s the first time and 164 s when I re-ran it), so I only trust the order of magnitude.
- **Reproduced:** I re-extracted every code block from this note into a fresh folder and re-ran seed 0 of all four models, the 25% run and the 60-epoch vanilla run. The validation accuracy after every epoch was **identical** to the first runs in all six cases. I did not re-run the other seeds, the 50% runs or the 60-epoch ResNet-18.
- **Not done:** no GPU, no 224-pixel runs, no tuning of any learning rate, and the validation photos were the only test set. See "What I did not run".

## What you need
Versions I used, in my existing Python environment on Windows 11 (**not a fresh virtual environment**; nothing was installed):

| Tool | Version |
|---|---|
| Python | 3.9.6 |
| torch | 2.8.0 (CPU build) |
| torchvision | 0.23.0 (CPU build) |

Two downloads, both public: the "hymenoptera" photo set from the PyTorch tutorials site (a 47,286,322-byte zip; I understand from memory that it is a small subset of ImageNet, which I did not check, so treat the photo licences as the ImageNet terms for study use), and the ResNet-18 weights `resnet18-f37072fd.pth` (44.7 MB), which torchvision downloaded (the file name carries a hash prefix, f37072fd; I did not check the hash myself). Everything lives in one folder outside the vault, `D:\python456\cnn-transfer`, because the vault holds notes and not project folders. I did not measure memory use; the runs were small.

## Steps
1. **Download and count the data.** The zip should hold 244 training and 153 validation photos in two folders per set.

   **`get_data.py`**
   ```python
   import urllib.request, zipfile, collections, os
   URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
   if not os.path.exists("hymenoptera_data.zip"): urllib.request.urlretrieve(URL, "hymenoptera_data.zip")
   print("zip bytes:", os.path.getsize("hymenoptera_data.zip"))
   z = zipfile.ZipFile("hymenoptera_data.zip"); z.extractall("data")
   count = collections.Counter(tuple(n.split("/")[1:3]) for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")))
   print(dict(count))
   ```
   Output: `zip bytes: 47286322` and `{('train', 'ants'): 123, ('train', 'bees'): 121, ('val', 'ants'): 70, ('val', 'bees'): 83}`. **What to look for:** nearly balanced classes, so "always say bee" gets 83 of 153 right (54.2%). That is the line any model must beat.

2. **Define the four models.** All four share everything else, so the only differences are what is being compared.
   - The **vanilla CNN** is deliberately plain: four blocks of ([[Convolution]], [[Batch Normalization]], [[Activation Function]] ReLU, 2 by 2 [[Pooling]]), then an average over the picture and one linear layer. It has 98,178 numbers.
   - **ResNet-18** is a real network of the kind that won image contests ([[ResNet]] and its [[Residual Connection]]s), with 11,177,538 numbers once its last layer is swapped for a new two-class layer. It comes in three kinds: random start ("from scratch"), pre-trained on ImageNet with everything frozen except the new layer ("frozen", 1,026 trainable numbers: 512 weights times 2 classes plus 2 biases), and pre-trained with everything allowed to change ("fine-tuned").
   - Learning rates: 0.001 for the three that start from random numbers or train only the new layer, and 0.0001 for fine-tuning, because a pre-trained network should be nudged gently ([[Learning Rate]]). I did not tune these.

   **`models.py`**
   ```python
   import torch.nn as nn
   from torchvision import models

   def vanilla_cnn():
       """A plain small CNN: four conv blocks (conv, batch norm, ReLU, 2x2 max pool), average the map, one linear layer."""
       def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
       return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2))

   def resnet18(pretrained, freeze=False):
       """ResNet-18 with the 1000-class head replaced by a fresh 2-class head. freeze=True trains only that head."""
       m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
       if freeze:
           for p in m.parameters(): p.requires_grad = False
       m.fc = nn.Linear(512, 2)                     # new layers are trainable by default
       return m

   METHODS = {                                      # name: (builder, learning rate)
       "vanilla CNN": (lambda: vanilla_cnn(), 1e-3),
       "ResNet-18 from scratch": (lambda: resnet18(False), 1e-3),
       "ResNet-18 pretrained, frozen": (lambda: resnet18(True, True), 1e-3),
       "ResNet-18 pretrained, fine-tuned": (lambda: resnet18(True), 1e-4),
   }
   ```
   I ran a quick count first: `vanilla CNN 98178 98178`, `ResNet-18 from scratch 11177538 11177538`, `ResNet-18 pretrained, frozen 11177538 1026` and `ResNet-18 pretrained, fine-tuned 11177538 11177538` (total and trainable numbers).

3. **Write the training script.** Every run: loads the pictures, makes the training pictures harder to memorise with a random crop and a random mirror flip ([[Data Augmentation]]), normalises them with the numbers ImageNet networks expect, trains with [[Adam Optimizer]] and [[Cross-Entropy Loss]] for a set number of epochs, and records the validation accuracy after each epoch. Details that matter:
   - **128 pixels:** my first version used 224-pixel pictures. One epoch took 48 s for the vanilla CNN and 76 s for the fine-tuned ResNet, so I estimated the full set of runs at about three hours. I switched to 128 pixels and decoded each file only once (a small cache), which gave about 10 s and 17 s per epoch.
   - **The frozen model keeps its backbone in evaluation mode.** Otherwise the batch-norm layers in the frozen part would still change their running averages while "frozen".
   - **`frac`** keeps only that share of each class in the training set (same pictures for every seed), for the "fewer pictures" test.

   **`run.py`**
   ```python
   import sys, time, json, random, torch, torch.nn as nn
   from PIL import Image
   from torchvision import datasets, transforms
   from models import METHODS

   torch.set_num_threads(8)
   name, seed, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
   frac = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0         # share of the training images to use
   torch.manual_seed(seed); random.seed(seed)
   norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # the numbers ImageNet models expect
   SIZE = 128                                                        # small pictures keep the CPU time reasonable
   train_tf = transforms.Compose([transforms.RandomResizedCrop(SIZE), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
   val_tf = transforms.Compose([transforms.CenterCrop(SIZE), transforms.ToTensor(), norm])
   cache = {}
   def load(path):                                                   # decode each file once, shrink the short side to 146 pixels
       if path not in cache:
           im = Image.open(path).convert("RGB"); w, h = im.size; s = 146 / min(w, h)
           cache[path] = im.resize((max(146, round(w * s)), max(146, round(h * s))), Image.BILINEAR)
       return cache[path]
   train_set = datasets.ImageFolder("data/hymenoptera_data/train", train_tf, loader=load)
   val_set = datasets.ImageFolder("data/hymenoptera_data/val", val_tf, loader=load)
   if frac < 1:                                                      # keep the same share of each class
       g = torch.Generator().manual_seed(0); keep = []
       for c in range(2):
           idx = [i for i, (_, y) in enumerate(train_set.samples) if y == c]
           keep += [idx[i] for i in torch.randperm(len(idx), generator=g)[:round(frac * len(idx))].tolist()]
       train_set = torch.utils.data.Subset(train_set, sorted(keep))
   train_dl = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
   val_dl = torch.utils.data.DataLoader(val_set, batch_size=64)

   build, lr = METHODS[name]
   model = build()
   params = [p for p in model.parameters() if p.requires_grad]
   total, trainable = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in params)
   opt = torch.optim.Adam(params, lr=lr)
   lossf = nn.CrossEntropyLoss()

   def set_mode():
       model.train()
       if name.endswith("frozen"): model.eval(); model.fc.train()   # keep the frozen backbone's batch-norm statistics fixed

   def val_acc():
       model.eval(); ok = 0
       with torch.no_grad():
           for x, y in val_dl: ok += (model(x).argmax(1) == y).sum().item()
       return ok / len(val_set)

   t0 = time.time(); hist = []
   for ep in range(epochs):
       set_mode(); tl = 0.0
       for x, y in train_dl:
           loss = lossf(model(x), y); opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(y)
       hist.append(val_acc())
       print(f"{name} seed {seed} epoch {ep+1}: train loss {tl/len(train_set):.4f}  val acc {hist[-1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
   json.dump({"method": name, "seed": seed, "frac": frac, "epochs": epochs, "val_acc": hist, "seconds": time.time() - t0,
              "params": total, "trainable": trainable, "n_train": len(train_set)}, open(f"results/{name.replace(' ', '_').replace(',', '')}_s{seed}_f{frac}_e{epochs}.json", "w"))
   ```

4. **Run the main comparison.** 4 models, 3 seeds, 15 epochs, the full training set, then the same with 25% and 50% of the pictures for three of the models (the from-scratch ResNet was left out of that test to save time).

   **`run_all.sh`**
   ```bash
   for seed in 0 1 2; do
     for m in "vanilla CNN" "ResNet-18 from scratch" "ResNet-18 pretrained, frozen" "ResNet-18 pretrained, fine-tuned"; do
       python -X utf8 run.py "$m" $seed 15 2>&1 | grep -v arn >> log_main.txt
     done
   done
   for frac in 0.25 0.5; do
     for seed in 0 1 2; do
       for m in "vanilla CNN" "ResNet-18 pretrained, frozen" "ResNet-18 pretrained, fine-tuned"; do
         python -X utf8 run.py "$m" $seed 15 $frac 2>&1 | grep -v arn >> log_frac.txt
       done
     done
   done
   echo done > all_done.txt
   ```

5. **Run the longer-training check.** I added this after I saw that the from-scratch models ended with a training loss of about 0.5, which means they had not even fitted their training pictures yet. Is the vanilla CNN just undertrained? Vanilla for 60 epochs, 3 seeds, and the from-scratch ResNet-18 for 60 epochs, one seed (it takes 11 minutes).

   **`run_long.sh`**
   ```bash
   for seed in 0 1 2; do python -X utf8 run.py "vanilla CNN" $seed 60 2>&1 | grep -v arn >> log_long.txt; done
   python -X utf8 run.py "ResNet-18 from scratch" 0 60 2>&1 | grep -v arn >> log_long.txt
   echo done > long_done.txt
   ```

6. **Summarise.** The script reads every result file and prints the tables.

   **`summarize.py`**
   ```python
   import json, glob, statistics as st, collections
   runs = collections.defaultdict(list)
   for f in sorted(glob.glob("results/*.json")):
       r = json.load(open(f)); runs[(r["method"], r["frac"], r["epochs"])].append(r)
   order = ["vanilla CNN", "ResNet-18 from scratch", "ResNet-18 pretrained, frozen", "ResNet-18 pretrained, fine-tuned"]
   ms = lambda v: f"{100*st.mean(v):.1f} (sd {100*st.pstdev(v):.1f})"
   print("majority class on val: ", round(100 * 83 / 153, 1))
   print("\nFULL TRAINING SET (244 images), 15 epochs, 3 seeds")
   for m in order:
       rs = runs.get((m, 1.0, 15), [])
       if not rs: continue
       fin = [r["val_acc"][-1] for r in rs]; last3 = [sum(r["val_acc"][-3:]) / 3 for r in rs]
       first90 = [next((i + 1 for i, a in enumerate(r["val_acc"]) if a >= 0.9), None) for r in rs]
       ep1 = [r["val_acc"][0] for r in rs]
       print(f"{m:34s} n={len(rs)} final {ms(fin)} | last3 {ms(last3)} | epoch1 {ms(ep1)} | per-seed final {[round(100*a,1) for a in fin]} | first epoch>=90%: {first90} | sec {st.mean(r['seconds'] for r in rs):.0f} | trainable {rs[0]['trainable']:,} of {rs[0]['params']:,}")
   print("\nFEWER TRAINING IMAGES (final-epoch val accuracy, mean (sd) over seeds)")
   for m in order:
       row = []
       for fr in (0.25, 0.5, 1.0):
           rs = runs.get((m, fr, 15), [])
           row.append(f"{fr}: n_train={rs[0]['n_train']} {ms([r['val_acc'][-1] for r in rs])} (n={len(rs)})" if rs else f"{fr}: -")
       print(f"{m:34s}", " | ".join(row))
   print("\nBEST-EPOCH (optimistic, uses val to pick) mean:")
   for m in order:
       rs = runs.get((m, 1.0, 15), [])
       if rs: print(f"{m:34s}", ms([max(r["val_acc"]) for r in rs]))

   print("\nLONGER TRAINING: 60 epochs, full training set")
   for m in order:
       rs = runs.get((m, 1.0, 60), [])
       if rs: print(f"{m:34s} n={len(rs)} final {ms([r['val_acc'][-1] for r in rs])} | last3 {ms([sum(r['val_acc'][-3:]) / 3 for r in rs])} | per-seed final {[round(100*r['val_acc'][-1],1) for r in rs]} | sec {st.mean(r['seconds'] for r in rs):.0f}")
   ```
   What it printed (shortened to the numbers used in this note; `sd` is the spread over the 3 seeds):
   ```text
   majority class on val:  54.2

   FULL TRAINING SET (244 images), 15 epochs, 3 seeds
   vanilla CNN                        final 70.8 (sd 3.9) | last3 73.2 (sd 1.9) | epoch1 53.2 (sd 7.9) | first epoch>=90%: [None, None, None]
   ResNet-18 from scratch             final 67.3 (sd 2.7) | last3 69.9 (sd 2.3) | epoch1 51.2 (sd 4.3) | first epoch>=90%: [None, None, None]
   ResNet-18 pretrained, frozen       final 88.9 (sd 1.4) | last3 88.4 (sd 0.7) | epoch1 78.4 (sd 6.2) | first epoch>=90%: [3, None, 2]
   ResNet-18 pretrained, fine-tuned   final 90.6 (sd 0.8) | last3 90.8 (sd 0.5) | epoch1 79.5 (sd 4.3) | first epoch>=90%: [2, 4, 3]

   FEWER TRAINING IMAGES (final-epoch val accuracy, mean (sd) over seeds)
   vanilla CNN                        0.25: n_train=61 61.9 (sd 2.0) | 0.5: n_train=122 68.0 (sd 2.7) | 1.0: n_train=244 70.8 (sd 3.9)
   ResNet-18 pretrained, frozen       0.25: n_train=61 86.9 (sd 1.8) | 0.5: n_train=122 87.8 (sd 1.5) | 1.0: n_train=244 88.9 (sd 1.4)
   ResNet-18 pretrained, fine-tuned   0.25: n_train=61 83.9 (sd 1.3) | 0.5: n_train=122 87.4 (sd 1.6) | 1.0: n_train=244 90.6 (sd 0.8)

   BEST-EPOCH (optimistic, uses val to pick) mean:
   vanilla CNN 77.3 | ResNet-18 from scratch 74.3 | frozen 90.0 | fine-tuned 92.4

   LONGER TRAINING: 60 epochs, full training set
   vanilla CNN                        n=3 final 74.1 (sd 1.9) | last3 74.4 (sd 0.7) | per-seed final [76.5, 71.9, 73.9]
   ResNet-18 from scratch             n=1 final 61.4 | last3 65.1
   ```
   **What to look for:** the two pre-trained models are already near 80% after one epoch and above 90% on some seeds within a few epochs, while the two from-scratch models never reached 90% on any seed (their best single epochs average 77.3% and 74.3%).

## How to read the numbers
- **The validation set is small.** 153 photos means each photo is worth 0.65 points, and the sampling noise of a score near 90% is about 2.4 points (the standard error is the square root of 0.9 times 0.1 over 153, my arithmetic). So 88.9% against 90.6% is **not** a clear win for fine-tuning, but 90% against 71% is far outside the noise.
- **Final epoch, not best epoch.** I report the accuracy after the last epoch. Picking each run's best epoch on the same validation photos would give higher numbers (77.3, 74.3, 90.0 and 92.4) but those are optimistically biased, so they are only in the printout.
- **Seeds:** 3 per setting, so the "sd" values are rough. The from-scratch models also bounce a lot from epoch to epoch, so one more or fewer epochs moves them by several points.
- **Fewer pictures:** at 25% (61 pictures) the frozen model (86.9%) was better than the fine-tuned model (83.9%), and the order flipped with the full data (88.9% against 90.6%). That is a pattern I would expect (fine-tuning needs enough data to avoid drifting) but with 3 seeds and these gaps I only call it "consistent with" that idea.

## What went wrong, and the fix
- **224-pixel runs were too slow.** My estimate was about three hours of CPU. Fixed by using 128 pixels and decoding each picture once. The pre-trained models probably lose a little accuracy at 128 pixels; I did not measure how much (my memory of the PyTorch tutorial is about 95% at 224 pixels for the frozen model, which I did not check or run).
- **The from-scratch models were undertrained at 15 epochs** (training loss about 0.5, against 0.05 to 0.2 for the pre-trained ones). The 60-epoch run was the fix for the worry that the comparison was unfair. It helped the vanilla CNN a little (70.8% to 74.1%) but not enough to matter.
- **I nearly overwrote results.** The result file names did not include the number of epochs, so my 60-epoch runs would have replaced the 15-epoch ones. I noticed before starting them, renamed the existing files to end with `_e15`, and added the epochs to the file name in `run.py`.
- **The vanilla CNN's validation accuracy jumps around** (for example 65% to 76% in neighbouring epochs). With batch norm and so few pictures its estimates are noisy. I report the last epoch and, in the printout, the mean of the last three.

## What I did not run
- **No GPU and no 224-pixel pictures**, so the absolute accuracies are lower than a tuned run would give.
- **No tuning.** Learning rates, epochs, augmentation and the vanilla CNN's design were chosen once and never searched. A better-designed small CNN may do better than my 70 to 74%, and a different learning rate may change the from-scratch ResNet.
- **The from-scratch ResNet-18 at 60 epochs has one seed**, and it was left out of the fewer-pictures test.
- **No separate test set.** The validation photos were the only evaluation, and I looked at them every epoch (but I did not use them to pick settings).
- **Only 3 seeds**, and no statistical test.
- **Only ants versus bees.** I did not check whether the pattern holds for other pictures, other pre-trained networks (for example a bigger ResNet or a [[Vision Transformer]]), or other transfer methods.
- **No look at what the networks see** (for example heat maps) and no look at which photos each model got wrong.

## Concepts used
- [[Transfer Learning]] · [[Fine-tuning]] · [[Convolutional Neural Network]] · [[Convolution]] · [[Pooling]] · [[ResNet]] · [[Residual Connection]] · [[Batch Normalization]] · [[Activation Function]] · [[Data Augmentation]] · [[Overfitting]] · [[Learning Rate]] · [[Adam Optimizer]] · [[Cross-Entropy Loss]] · [[Image Classification]] · [[Pre-training]] · [[PyTorch]] · [[ResNet 2015]]
- [[Tiny Object Detector]]
## Next ideas
- Run at 224 pixels on a machine that can afford it, and see how many points the pre-trained models gain.
- Give the vanilla CNN a fair fight: tune its learning rate, width and augmentation, and try more epochs with a learning-rate schedule.
- Add a bigger pre-trained model and a [[Vision Transformer]] to the comparison.
- Look at the photos each model gets wrong, and at heat maps of what it looks at.
- Fine-tune only the last block of the ResNet, between "frozen" and "fine-tuned".

## Flashcards
#flashcards/projects

#flashcards/projects/reverse #flashcards/topic/project-deep-learning Transfer learning (this project):::Starting from a network pre-trained on a big dataset (ImageNet) instead of from random numbers, then training only a new last layer (frozen) or everything gently (fine-tuned).

#flashcards/projects/concept #flashcards/topic/project-deep-learning What was the task and the data in the project?
?
Classify ants versus bees from 244 training photos (123 ants, 121 bees) and 153 validation photos (70 ants, 83 bees), at 128 by 128 pixels on a CPU. Always guessing "bee" scores 54.2%.

#flashcards/projects/example #flashcards/topic/project-deep-learning What were the 15-epoch validation accuracies of the four models (mean of 3 seeds)?::Vanilla CNN 70.8%, ResNet-18 from scratch 67.3%, ResNet-18 pre-trained and frozen 88.9%, ResNet-18 pre-trained and fine-tuned 90.6%. Standard deviations over seeds: 3.9, 2.7, 1.4 and 0.8.

#flashcards/projects/concept #flashcards/topic/project-deep-learning How many numbers did each of the four models train?::Vanilla CNN 98,178. ResNet-18 from scratch 11,177,538. ResNet-18 frozen 1,026 (512 times 2 weights plus 2 biases). ResNet-18 fine-tuned 11,177,538.

#flashcards/projects/concept #flashcards/topic/project-deep-learning Did a big architecture alone beat the small vanilla CNN?
?
No. The same ResNet-18 from random weights scored 67.3% against 70.8% for the vanilla CNN. The gain came from the ImageNet pre-training, not from the architecture.

#flashcards/projects/example #flashcards/topic/project-deep-learning How did the models do with only 61 training photos (25%)?::Vanilla CNN 61.9%, frozen pre-trained 86.9%, fine-tuned pre-trained 83.9%. The frozen model on 61 photos beat the vanilla CNN on all 244 (70.8%) by about 16 points.

#flashcards/projects/concept #flashcards/topic/project-deep-learning Did training the from-scratch models longer close the gap?
?
Not here. With 60 epochs the vanilla CNN reached 74.1% (3 seeds), and ResNet-18 from scratch scored 61.4% (one seed). The pre-trained models were at 88.9% and 90.6% after 15 epochs. The from-scratch models were still undertrained (training loss about 0.38 after 60 epochs).

#flashcards/projects/concept #flashcards/topic/project-deep-learning Is 88.9% (frozen) versus 90.6% (fine-tuned) a real difference?
?
Not clearly. The validation set has 153 photos, one photo is 0.65 points, and the sampling noise near 90% is about 2.4 points (my arithmetic). The 18-to-23-point gap to the from-scratch models is far outside that noise.

#flashcards/projects/concept #flashcards/topic/project-deep-learning Why did the frozen model keep its backbone in evaluation mode?
?
Batch-norm layers update their running averages in training mode, so a "frozen" backbone would still change. Evaluation mode keeps those statistics fixed, and only the new last layer trains.

#flashcards/projects/concept #flashcards/topic/project-deep-learning Why was the project run at 128 pixels, and what is the cost?
?
At 224 pixels the runs would have taken about three hours by my estimate (48 s per epoch for the vanilla CNN, 76 s for the fine-tuned ResNet). At 128 pixels with each picture decoded once it was about 10 s and 17 s. The cost is probably lower accuracy for the pre-trained models, which I did not measure.

#flashcards/projects/cloze #flashcards/topic/project-deep-learning A model that always guesses the most common class scores ==54.2%== on the validation photos (83 of 153), the line every model must beat.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What did this project not run?
?
No GPU or 224-pixel pictures, no tuning of learning rates or the vanilla CNN, only one seed for ResNet-18 at 60 epochs, no separate test set, only 3 seeds, only ants versus bees, no other pre-trained networks, and no look at heat maps or at the photos each model got wrong.
