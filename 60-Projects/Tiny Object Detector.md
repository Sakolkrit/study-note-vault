---
tags: [project, computer-vision, object-detection, faster-rcnn, pytorch, coco, cpu, multi-class]
status: done
stack: Python 3.11 in its own venv, PyTorch 2.8.0 (CPU), torchvision 0.23.0, pycocotools 2.0.11, matplotlib 3.11.2, Pillow 13.0.0, NumPy 2.4.6
goal: train a small object detector on an ordinary laptop CPU to find three kinds of things (cat, dog, bus) in photos, measure how good it is with the standard detection score, and find out where it fails
---

# Tiny Object Detector

> **In one line:** take a small [[Object Detection]] network ([[Faster R-CNN 2015]] with a light MobileNetV3 backbone), teach it to draw boxes around **cats, dogs and buses** using 1,200 real photos on a laptop **without a GPU**, and measure what it finds, what it confuses and what it misses.

![[tiny-detector-examples.png]]
*Four validation photos with the detector's boxes (red = cat, blue = dog, green = bus, with its score) and the true boxes in white. Left two: correct. Third: a dog called a cat (0.90). Right: three sled dogs, none found. Photos from COCO val2017, taken from Flickr; COCO's file gives each photo's licence and Flickr address but no photographer name. From left to right: COCO file 000000108244 (CC Attribution-ShareAlike, `farm1.staticflickr.com/49/165710819_fb98979f1e_z.jpg`), 000000568290 (CC Attribution-ShareAlike, `farm8.staticflickr.com/7342/9305416099_b62431c75b_z.jpg`), 000000182805 (CC Attribution-ShareAlike, `farm7.staticflickr.com/6218/6309322560_cf4708d8dc_z.jpg`), 000000107226 (CC Attribution, `farm3.staticflickr.com/2139/2185708967_4da3c7fb32_z.jpg`). Boxes and labels added by me, so the picture is a derivative and is shared under the same licence terms (the Attribution-ShareAlike one is the stricter).*

## Goal
Done means: (1) a detector trained from an ImageNet-pretrained backbone only (not a detector that already knows these classes) on a 3-class subset of COCO, (2) its COCO score (mAP) on 240 photos it never trained on, per class, (3) a confusion table showing which class is mistaken for which, (4) a measure of where it fails (by object size), and (5) the project in **its own virtual environment** with pinned versions. **Not a state-of-the-art detector:** 8 epochs, one run, one small data set.

## Result in short
- **Data:** a subset of COCO 2017, three classes. Train: 1,200 photos (400 per class, picked at random with a fixed seed, at most 4 target boxes each), 1,510 boxes, from COCO's train set. Validation: 240 photos (80 per class), 305 boxes, from COCO's separate validation set (cats 90, dogs 102, buses 113 boxes).
- **Model:** Faster R-CNN with a MobileNetV3-Large backbone and feature pyramid, 320-pixel version. Only the backbone had pre-trained weights (ImageNet); the detection heads started random.
- **Cost:** training 8 epochs took **2,809 s (46.8 minutes)** on my laptop's CPU (4 threads, with another program also using the CPU), with a summed training loss falling from 1.220 to 0.473. A photo takes about **133 to 157 ms** to detect (two runs).
- **Score on 240 unseen photos:** **mAP 0.407** (average over IoU 0.5 to 0.95), **0.695** at IoU 0.5 and 0.424 at IoU 0.75. Per class: cat 0.452, dog 0.359, bus 0.409 (at IoU 0.5: cat 0.808, dog 0.669, bus 0.607).
- **At score 0.5** (the detector is "sure"): of 305 true boxes, 206 (67.5%, my arithmetic) were found with the right class: cats 72 of 90, dogs 66 of 102, buses 68 of 113. There were 64 extra boxes that matched no true box.
- **Cat and dog are rarely swapped:** 3 cats were called dogs and 7 dogs called cats. The bigger problem is **misses**: 15 cats, 29 dogs and 45 buses (40% of the buses) got no box at all. No bus was ever called an animal and no animal a bus.
- **Size is the main problem:** true boxes under 2% of the photo were found 7.3% of the time (41 boxes), those of 2% to 15% 59.8% (102 boxes), and the large ones (over 15%) 87.7% (162 boxes).
- **Not done:** one run, no repeated seeds or error bars, no longer training (the loss was still falling), no comparison with a detector trained from scratch or one already trained on COCO, no photos of my own, no webcam test.

## What you need
- A Windows laptop with Python 3.11 (`py -3.11`) and about 1 GB free (data 0.5 GB, venv 0.6 GB). No GPU, no account. Internet to download COCO annotations (253 MB zip), the 1,440 photos (about 220 MB) and the backbone weights (about 20 MB).
- **Its own virtual environment** (so nothing here touches your other Python). Commands (PowerShell, in an empty project folder):

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install pycocotools matplotlib
```

- The versions I got: torch 2.8.0+cpu, torchvision 0.23.0+cpu, pycocotools 2.0.11, matplotlib 3.11.2, Pillow 13.0.0, NumPy 2.4.6. Below, run every script with `.venv\Scripts\python.exe`.

## Steps
1. **Choose and download the data.** `prep_data.py` downloads COCO's annotation zip (only two files are used), picks 400 train and 80 validation photos per class that contain at most 4 boxes of the three classes and boxes of at least 20×20 pixels (to skip specks), downloads them, and writes `data/subset.json`. Label 0 is background. About 10 minutes, mostly download.

```python
# file: prep_data.py
"""Pick a small 3-class subset of COCO (cat, dog, bus), download the images, write one json."""
import json, random, os, sys, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor

CLASSES = ["cat", "dog", "bus"]
N_TRAIN, N_VAL = 400, 80          # images per class
random.seed(0)

def pick(split, per_class):
    d = json.load(open(f"data/annotations/instances_{split}2017.json"))
    cid = {c["name"]: c["id"] for c in d["categories"]}
    inv = {cid[n]: i + 1 for i, n in enumerate(CLASSES)}   # label 0 = background
    imgs = {i["id"]: i for i in d["images"]}
    boxes = {}
    for a in d["annotations"]:
        if a["category_id"] in inv and not a["iscrowd"]:
            x, y, w, h = a["bbox"]
            if w >= 20 and h >= 20:
                boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h, inv[a["category_id"]]])
    chosen, out = set(), []
    for lab in (1, 2, 3):
        # images that contain this class and at most 4 boxes of the three classes
        pool = sorted(i for i, b in boxes.items()
                      if any(t[4] == lab for t in b) and len(b) <= 4 and i not in chosen)
        random.shuffle(pool)
        for i in pool[:per_class]:
            chosen.add(i)
            im = imgs[i]
            out.append({"id": i, "file": im["file_name"], "w": im["width"], "h": im["height"],
                        "license": im["license"], "flickr": im["flickr_url"], "boxes": boxes[i]})
    return out

def fetch(url, path, tries=5):
    """Download url to path, trying again when the connection stalls."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 study-vault"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                while chunk := r.read(1 << 16): f.write(chunk)
            return
        except OSError as e:
            print("retry", k + 1, url, e, flush=True)
    raise RuntimeError("could not download " + url)

def get(split, rec):
    path = f"data/images/{rec['file']}"
    if not os.path.exists(path):
        fetch(f"http://images.cocodataset.org/{split}2017/{rec['file']}", path)

os.makedirs("data/images", exist_ok=True)
if not os.path.exists("data/annotations/instances_val2017.json"):       # 253 MB zip, only two files are used
    fetch("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "data/ann.zip")
    with zipfile.ZipFile("data/ann.zip") as z:
        for n in ("annotations/instances_train2017.json", "annotations/instances_val2017.json"): z.extract(n, "data")
    os.remove("data/ann.zip")
meta = {"classes": CLASSES}
for split, n in (("train", N_TRAIN), ("val", N_VAL)):
    recs = pick(split, n)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda r: get(split, r), recs))
    meta[split] = recs
    print(split, len(recs), "images,", sum(len(r["boxes"]) for r in recs), "boxes")
json.dump(meta, open("data/subset.json", "w"))
```

```text
train 1200 images, 1510 boxes
val 240 images, 305 boxes
```

2. **Train.** `train.py` builds the detector (`weights=None` for the detector, `weights_backbone="IMAGENET1K_V1"` for the backbone only), sets 4 classes (background plus three), and trains 8 epochs with batches of 8, SGD (momentum 0.9, weight decay 1e-4) and a "one cycle" learning rate rising to 0.02 and falling. The only augmentation is a random left-right flip. I made these choices once and did not tune them. About 47 minutes.
   - The three numbers `rpn_pre_nms_top_n_train=300`, `rpn_post_nms_top_n_train=100` and `box_batch_size_per_image=32` limit how many candidate boxes the training looks at per photo. I lowered them for speed (see "What went wrong").

```python
# file: train.py
"""Train a small Faster R-CNN (MobileNetV3 backbone, ImageNet weights only) on the cat/dog/bus subset."""
import json, sys, time, random
import numpy as np, torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0     # >0: use only that many train images (quick test)
SEED = 0
torch.set_num_threads(4); torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
meta = json.load(open("data/subset.json"))

class DS(torch.utils.data.Dataset):
    def __init__(self, recs, train):
        self.recs, self.train = recs, train
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        img = TF.to_tensor(Image.open(f"data/images/{r['file']}").convert("RGB"))
        b = torch.tensor([x[:4] for x in r["boxes"]], dtype=torch.float32)
        lab = torch.tensor([x[4] for x in r["boxes"]], dtype=torch.int64)
        if self.train and random.random() < 0.5:
            img = img.flip(-1); w = img.shape[-1]
            b = torch.stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]], 1)
        return img, {"boxes": b, "labels": lab}

def build():
    m = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone="IMAGENET1K_V1", num_classes=4,
                                          rpn_pre_nms_top_n_train=300, rpn_post_nms_top_n_train=100, box_batch_size_per_image=32)
    return m

if __name__ == "__main__":
    train = meta["train"][:LIMIT] if LIMIT else meta["train"]
    dl = torch.utils.data.DataLoader(DS(train, True), batch_size=8, shuffle=True, collate_fn=lambda b: tuple(zip(*b)), num_workers=0)
    m = build(); m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=1e-4)
    total = EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.02, total_steps=total, pct_start=0.1)
    log = []
    t0 = time.time()
    for ep in range(EPOCHS):
        s, n = 0.0, 0
        for imgs, tg in dl:
            loss = sum(m(list(imgs), list(tg)).values())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            s += loss.item(); n += 1
        log.append({"epoch": ep + 1, "loss": s / n, "sec": time.time() - t0})
        print(f"epoch {ep+1}/{EPOCHS} loss {s/n:.3f} elapsed {time.time()-t0:.0f}s", flush=True)
    torch.save(m.state_dict(), "detector.pt")
    json.dump(log, open("train_log.json", "w"))
```

```text
epoch 1/8 loss 1.220 elapsed 390s
epoch 2/8 loss 1.055 elapsed 774s
epoch 3/8 loss 0.954 elapsed 1131s
epoch 4/8 loss 0.862 elapsed 1468s
epoch 5/8 loss 0.771 elapsed 1844s
epoch 6/8 loss 0.666 elapsed 2177s
epoch 7/8 loss 0.555 elapsed 2496s
epoch 8/8 loss 0.473 elapsed 2809s
```

3. **Score it.** `evaluate.py` runs the saved detector on the 240 validation photos, computes COCO's mAP with `pycocotools`, the per-class average precision, and a confusion table at score 0.5: each true box is matched to the predicted box with the highest overlap, and counts as found if the overlap (IoU) is at least 0.5; the column says which class the detector gave it, and "none" means no box. About 1 minute.

```python
# file: evaluate.py
"""Evaluate detector.pt on the 240 validation images: COCO mAP, per-class AP, confusion at score 0.5, speed."""
import json, time, io, contextlib
import numpy as np, torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import train as T

torch.set_num_threads(4)
meta = T.meta; names = ["background"] + meta["classes"]
m = T.build(); m.load_state_dict(torch.load("detector.pt")); m.eval()
ds = T.DS(meta["val"], False)

# ground truth in COCO format
gt = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(names) if i]}
k = 1
for j, r in enumerate(meta["val"]):
    gt["images"].append({"id": j, "width": r["w"], "height": r["h"]})
    for x0, y0, x1, y1, c in r["boxes"]:
        gt["annotations"].append({"id": k, "image_id": j, "category_id": c, "bbox": [x0, y0, x1 - x0, y1 - y0],
                                  "area": (x1 - x0) * (y1 - y0), "iscrowd": 0}); k += 1
dets, preds = [], []
t0 = time.time()
with torch.no_grad():
    for j in range(len(ds)):
        img, tg = ds[j]
        o = m([img])[0]
        preds.append((o["boxes"], o["scores"], o["labels"]))
        for b, s, l in zip(o["boxes"].tolist(), o["scores"].tolist(), o["labels"].tolist()):
            dets.append({"image_id": j, "category_id": l, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
sec = (time.time() - t0) / len(ds)
print(f"inference {sec*1000:.0f} ms per image, {len(dets)} detections")

with contextlib.redirect_stdout(io.StringIO()):
    G = COCO(); G.dataset = gt; G.createIndex()
    D = G.loadRes(dets); E = COCOeval(G, D, "bbox"); E.evaluate(); E.accumulate()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    E.summarize()
st = E.stats
print(f"mAP@[.5:.95] {st[0]:.3f}   mAP@.5 {st[1]:.3f}   mAP@.75 {st[2]:.3f}")
res = {"map": st[0], "map50": st[1], "map75": st[2], "ms_per_image": sec * 1000}
# per-class AP (IoU .5:.95 and .5)
p = E.eval["precision"]   # T x R x K x A x M
for ci, n in enumerate(names[1:]):
    a = p[:, :, ci, 0, 2]; a50 = p[0, :, ci, 0, 2]
    f = lambda x: float(np.mean(x[x > -1])) if (x > -1).any() else float("nan")
    print(f"  {n:4s} AP {f(a):.3f}  AP50 {f(a50):.3f}")
    res[n] = {"ap": f(a), "ap50": f(a50)}

# confusion at score >= 0.5, IoU >= 0.5: rows = true class, cols = predicted class (0 = missed / no box)
from torchvision.ops import box_iou
conf = np.zeros((4, 4), int); fp = 0
for j, r in enumerate(meta["val"]):
    gb = torch.tensor([x[:4] for x in r["boxes"]], dtype=torch.float32); gl = [x[4] for x in r["boxes"]]
    b, s, l = preds[j]; keep = s >= 0.5; b, l = b[keep], l[keep]
    used = set()
    if len(b):
        iou = box_iou(gb, b)
        for gi in range(len(gl)):
            best = int(iou[gi].argmax()); 
            if iou[gi, best] >= 0.5: conf[gl[gi], int(l[best])] += 1; used.add(best)
            else: conf[gl[gi], 0] += 1
        fp += len(b) - len(used)
    else:
        for g in gl: conf[g, 0] += 1
print("confusion (rows true, cols predicted, col 0 = missed):")
print("      " + "  ".join(f"{n:>5s}" for n in ["none"] + names[1:]))
for i in range(1, 4): print(f"{names[i]:>5s} " + "  ".join(f"{v:5d}" for v in conf[i]))
print("extra boxes with no matching true box:", fp)
res["confusion"] = conf.tolist(); res["extra_boxes"] = fp
json.dump(res, open("eval.json", "w"), indent=1)
torch.save(preds, "val_preds.pt")
```

```text
inference 133 ms per image, 1112 detections
mAP@[.5:.95] 0.407   mAP@.5 0.695   mAP@.75 0.424
  cat  AP 0.452  AP50 0.808
  dog  AP 0.359  AP50 0.669
  bus  AP 0.409  AP50 0.607
confusion (rows true, cols predicted, col 0 = missed):
       none    cat    dog    bus
  cat    15     72      3      0
  dog    29      7     66      0
  bus    45      0      0     68
extra boxes with no matching true box: 64
```

4. **Find where it fails.** `analyse.py` groups the true boxes by their size as a share of the photo and counts how many were found with the right class at score 0.5, and draws the picture at the top (only photos whose COCO licence field allows reuse; the four photos were picked by me after looking at the list of mistakes: two clean successes, a dog called a cat, and a photo with three missed dogs).

```python
# file: analyse.py
"""Misses by box size, and a picture of detections on a few validation images."""
import json, torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou
import train as T

meta = T.meta; names = ["", "cat", "dog", "bus"]
preds = torch.load("val_preds.pt")
# 1. found / missed by size of the true box (share of the 640x480 image area)
rows = []
for j, r in enumerate(meta["val"]):
    b, s, l = preds[j]; k = s >= 0.5; b, l = b[k], l[k]
    for x0, y0, x1, y1, c in r["boxes"]:
        area = (x1 - x0) * (y1 - y0) / (r["w"] * r["h"])
        hit = False
        if len(b):
            iou = box_iou(torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32), b)[0]
            i = int(iou.argmax()); hit = bool(iou[i] >= 0.5 and int(l[i]) == c)
        rows.append((area, hit))
rows = np.array(rows)
for lo, hi, n in ((0, 0.02, "small  (< 2% of image)"), (0.02, 0.15, "medium (2% to 15%)"), (0.15, 1.01, "large  (> 15%)")):
    sel = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    print(f"{n}: {int(sel.sum()):3d} true boxes, correct class and IoU>=0.5 at score 0.5: {rows[sel,1].mean()*100:.1f}%")

# 2. picture: 2 correct examples and 2 failures, from images whose Flickr licence allows reuse (CC BY, CC BY-SA, no known copyright, US Gov)
ok = {4, 5, 7, 8}
col = {1: "#d62728", 2: "#1f77b4", 3: "#2ca02c"}
FONT = ImageFont.truetype("arial.ttf", 22)
def draw(j):
    r = meta["val"][j]; im = Image.open(f"data/images/{r['file']}").convert("RGB"); d = ImageDraw.Draw(im)
    for x0, y0, x1, y1, c in r["boxes"]:
        d.rectangle([x0, y0, x1, y1], outline="white", width=2)
    b, s, l = preds[j]
    for bb, ss, ll in zip(b.tolist(), s.tolist(), l.tolist()):
        if ss >= 0.5:
            d.rectangle(bb, outline=col[ll], width=4); d.text((bb[0] + 6, bb[1] + 4), f"{names[ll]} {ss:.2f}", fill="white", font=FONT, stroke_width=2, stroke_fill=col[ll])
    return im
cands = [j for j, r in enumerate(meta["val"]) if r["license"] in ok]
def status(j):
    r = meta["val"][j]; b, s, l = preds[j]; k = s >= 0.5; b, l = b[k], l[k]
    good = 0
    for x0, y0, x1, y1, c in r["boxes"]:
        if len(b):
            iou = box_iou(torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32), b)[0]; i = int(iou.argmax())
            good += int(iou[i] >= 0.5 and int(l[i]) == c)
    return good, len(r["boxes"]), len(b)
good = [j for j in cands if status(j)[0] == status(j)[1] and status(j)[2] == status(j)[1]]
bad = [j for j in cands if status(j)[0] < status(j)[1]]
byfile = {r["file"]: j for j, r in enumerate(meta["val"])}
pick = [byfile[f] for f in ("000000108244.jpg", "000000568290.jpg", "000000182805.jpg", "000000107226.jpg")]
print("picked", [(meta["val"][j]["file"], meta["val"][j]["license"], status(j)) for j in pick])
tiles = [draw(j).resize((420, int(420 * meta["val"][j]["h"] / meta["val"][j]["w"]))) for j in pick]
H = max(t.size[1] for t in tiles); W = sum(t.size[0] for t in tiles) + 10 * (len(tiles) - 1)
canvas = Image.new("RGB", (W, H), "white"); x = 0
for t in tiles: canvas.paste(t, (x, 0)); x += t.size[0] + 10
canvas.save("detections.png")
json.dump([{"file": meta["val"][j]["file"], "license": meta["val"][j]["license"], "flickr": meta["val"][j]["flickr"]} for j in pick], open("picked.json", "w"), indent=1)
```

```text
small  (< 2% of image):  41 true boxes, correct class and IoU>=0.5 at score 0.5: 7.3%
medium (2% to 15%): 102 true boxes, correct class and IoU>=0.5 at score 0.5: 59.8%
large  (> 15%): 162 true boxes, correct class and IoU>=0.5 at score 0.5: 87.7%
picked [('000000108244.jpg', 5, (1, 1, 1)), ('000000568290.jpg', 5, (1, 1, 1)), ('000000182805.jpg', 5, (0, 2, 1)), ('000000107226.jpg', 4, (0, 3, 0))]
```

5. **Read it.**
   - The mAP of 0.407 is an average over strict and loose overlaps; at the loose IoU of 0.5 it is 0.695. The gap shows that the boxes are often roughly right but not tight.
   - In the confusion table the wrong class is rare, the empty first column is large. The network's weak point is **not seeing** the object, not mislabelling it.
   - The size table gives the reason in this data: small objects are mostly not found. Many of the 45 missed buses are probably small or far away (not checked bus by bus).

## What went wrong, and the fix
- **The first training speed was far too slow.** One epoch on 64 photos took 119 s, about 1 s per photo, which would be about 20 minutes per epoch. I timed two batches of 8 photos in several set-ups: 13.0 s with 8 threads and 9.2 s with 4 threads (backbone partly frozen, 3 trainable layers). Fewer threads was faster, probably because the laptop was also busy (another program of mine was using about half of the CPU, and Windows reported 37% to 50% load before I started), but I did not investigate.
- **Cutting the candidate boxes** (the three numbers in step 2) brought it to 2.7 s per batch of 8 (0.34 s per photo), so an epoch took about 6.5 minutes. I did not measure what that costs in accuracy, so a better score may be possible with the full settings. A lighter detector (SSDlite) timed at 1.28 s per batch of 8; I did not train it.
- **My first picture picked a weak "failure".** It showed a blurry cat that was found, in a photo with two true boxes where I could not see the second one (so it was not a clear lesson). I replaced it with two clearer failures (a dog called a cat, three missed dogs) and made the score labels larger.
- **A caution about the confusion table.** One predicted box can be the best match for two true boxes (for example two dogs next to each other); I did not remove such double matches, so the counts are approximate near crowded dogs.
- **A fresh virtual environment failed the first time I tried the install commands from this note.** A new venv comes with an old pip. With it, the PyTorch install stopped at `typing-extensions` with "Could not find a version that satisfies the requirement flit_core". My own venv had worked because I had upgraded pip first, and I had forgotten to write that step down. The first line of the install block is now `pip install --upgrade pip`. The re-run check below used the corrected commands.
- **The data download stalled in my first fresh re-run, twice.** The server sent the headers of the 253 MB annotation zip and then no data: reading it in 1 MB pieces with Python's `urllib` timed out, while 64 KB pieces ran at about 10 MB/s and `curl` also worked at about 12 MB/s (I tested each for 12 to 20 seconds). `prep_data.py` now reads 64 KB at a time and tries each download up to 5 times. I do not know why the large reads stall, so this is a workaround, not an explanation.
- **Nothing else broke.** The scripts ran to the end, and rerunning `evaluate.py` and `analyse.py` gave identical scores.

## What I did not run
- **More than one training run.** One seed. The differences between classes (cat 0.452, dog 0.359, bus 0.409) come from 90 to 113 boxes each and may not be real differences; no error bars ([[Confidence Interval]], [[Bootstrap]] would give them).
- **Longer training.** The loss was still falling at epoch 8 (0.555 to 0.473 in the last epoch), so the score probably is not the best this model can do (not tested).
- **A detector from scratch, or a COCO-pretrained detector.** The backbone's ImageNet pre-training helps (from memory, ImageNet has cat, dog and bus-like classes); I did not train a backbone from random weights to see how much. A detector already trained on COCO would know these classes, which is why I did not use one.
- **More augmentation**, other learning rates, bigger input size, other detectors (YOLO, SSD).
- **Photos of my own, a webcam or a video.** Only COCO photos, 640×480 at most.
- **Speed on video.** 133 to 157 ms per photo is about 6 to 7 photos per second on my laptop; I did not run a video.
- **A GPU.** None here.
- **Tuning the score threshold** (0.5 was fixed in the script before I saw results) and the non-maximum-suppression setting (library defaults, not printed, from memory 0.5).
- **COCO licences in detail.** Only the four photos in the picture were checked for their licence field; the 1,440 downloaded photos stay on my disk and are not in the vault.

## Re-run check
The four scripts were extracted from this note into a fresh, empty folder and run in order with a **new virtual environment** built from the install commands above (the pip fix included): the data download, the training, the scoring and the picture script. Every output matched the output block above, including the eight training losses to three decimals, the mAP numbers, the confusion table and the size table (timings are ignored in the comparison). So with the same seed on this CPU the result is repeatable. The first two attempts failed for the two reasons in "What went wrong" (old pip, stalled download) and were repeated after the fixes. The full re-run took more than 2.5 hours, against 47 minutes for the first training, because other programs were using the CPU. Not tested: another machine, other library versions, a GPU.

## Concepts used
- [[Object Detection]] · [[Faster R-CNN 2015]] · [[Convolutional Neural Network]] · [[Transfer Learning]] · [[Fine-tuning]] · [[Data Augmentation]] · [[Intersection over Union]] · [[Mean Average Precision]] · [[Non-Maximum Suppression]] · [[Confusion Matrix]] · [[Precision]] · [[Precision-Recall Curve]] · [[Evaluation Metrics]] · [[Image Classification]] · [[Confidence Interval]] · [[Bootstrap]] · [[MobileNet]]

## Next ideas
- Train 20 epochs and see whether small objects improve; or give the network bigger photos (min size 480).
- Repeat with 5 seeds and put error bars on the three class scores ([[Error Bars Lab]] method).
- Compare with the same network with a random backbone, to measure what ImageNet pre-training gives.
- Add a fourth class, or take photos of three things on your desk and train on them.
- Run a webcam demo and measure the frames per second.

## Sources
- My own code and runs (versions above). Data: COCO 2017 (annotation file says "COCO 2017 Dataset", contributor "COCO Consortium"; licence table read from the file: ids 4 and 5 are Attribution and Attribution-ShareAlike). COCO's paper, the licence versions (2.0), ImageNet's classes, and torchvision's defaults are from memory; I did not read torchvision's documentation for this note.
- The picture was made by me from four COCO validation photos (see its caption).

## Flashcards
#flashcards/projects

#flashcards/projects/image #flashcards/topic/project-deep-learning ![[tiny-detector-examples.png]]
What do the coloured and white boxes show, and what do the four photos teach?
?
White = true boxes, coloured = the detector's boxes with scores (red cat, blue dog, green bus). The first two photos are found correctly; in the third a dog is called a cat (0.90); in the fourth three sled dogs get no box. COCO val2017 photos (Flickr, CC Attribution or Attribution-ShareAlike as listed in COCO's file).

#flashcards/projects/concept #flashcards/topic/project-deep-learning What was the setup of Tiny Object Detector?
?
A Faster R-CNN with a MobileNetV3-Large backbone (ImageNet weights only, random heads), trained 8 epochs on 1,200 COCO photos (400 each of cat, dog, bus) on a laptop CPU in its own venv, and tested on 240 other photos (80 each).

#flashcards/projects/concept #flashcards/topic/project-deep-learning How good was the Tiny Object Detector, and how long did it take?
?
mAP 0.407 over IoU 0.5 to 0.95, 0.695 at IoU 0.5 and 0.424 at IoU 0.75, on 240 unseen photos. Training took 2,809 s (46.8 minutes) on 4 CPU threads, loss 1.220 to 0.473; detection takes about 133 to 157 ms per photo. One run, no error bars.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What were the per-class scores of the Tiny Object Detector?
?
mAP over IoU 0.5 to 0.95: cat 0.452, dog 0.359, bus 0.409. At IoU 0.5: cat 0.808, dog 0.669, bus 0.607. Each class has only 90 to 113 true boxes and there is one run, so the order of the classes is not firmly established.

#flashcards/projects/concept #flashcards/topic/project-deep-learning Did the Tiny Object Detector mix up cats and dogs, or miss objects?
?
Mostly it missed them. At score 0.5, of 305 true boxes, 206 were found with the right class; only 3 cats were called dogs and 7 dogs called cats, no bus was called an animal. Misses: 15 cats, 29 dogs and 45 buses (40% of buses). It also made 64 extra boxes.

#flashcards/projects/concept #flashcards/topic/project-deep-learning How did the size of the object change what the Tiny Object Detector found?
?
At score 0.5, found with the right class and IoU at least 0.5: small boxes (under 2% of the photo) 7.3% of 41; medium (2% to 15%) 59.8% of 102; large (over 15%) 87.7% of 162. Small objects are the main weakness here.

#flashcards/projects/concept #flashcards/topic/project-deep-learning Why was the first training of the Tiny Object Detector too slow, and what fixed it?
?
About 1 s per photo (13.0 s per batch of 8 with 8 threads, 9.2 s with 4). Lowering the candidate boxes per photo (300, 100 and 32 instead of the defaults) gave 2.7 s per batch of 8, about 6.5 minutes per epoch. The accuracy cost of that was not measured.

#flashcards/projects/concept #flashcards/topic/project-deep-learning Why did Tiny Object Detector not start from a detector already trained on COCO?
?
That detector already knows cat, dog and bus, so the test would not show what the small data teaches. Only the backbone had ImageNet weights; the detection heads started random. How much the ImageNet start helps was not measured.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What did Tiny Object Detector not run?
?
More than one seed (no error bars); longer training (loss still falling); training a backbone from scratch or a COCO-pretrained detector; more augmentation or other detectors; photos of my own or video; threshold and NMS tuning. Only COCO photos on a CPU.

#flashcards/projects/cloze #flashcards/topic/project-deep-learning The Tiny Object Detector used a ==Faster R-CNN== with a ==MobileNetV3== backbone and only ==ImageNet== weights, trained on a laptop with no ==GPU==.
