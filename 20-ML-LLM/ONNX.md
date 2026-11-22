---
tags: [ONNX, ONNX-Runtime, model-export, deployment, inference, MLOps, ML]
status: growing
level: 4
tier: 2
---

# ONNX

ONNX (Open Neural Network Exchange) is **a common file format for trained models**. You train a model in one tool (scikit-learn, PyTorch, TensorFlow), save it as an ONNX file, and run that file anywhere an ONNX engine exists, without the original tool.

## 🟢 1. Simple (age 12)
You write a school report in one program and save it as a **PDF**. Now anyone, on any computer, with any reader, can open it and see exactly the same pages, even if they do not have your program. **ONNX is like a PDF for trained models.** The model is made in one program and saved in a format that many other programs and machines can read and run.

**Example:** a friend trains a price-guessing model in Python with a big library. Your app is written in another language and runs on a small server with none of those libraries. Your friend saves the model as an ONNX file; your app loads it with a small ONNX engine and gets the same answers.

## 🔵 2. How it works
1. **Train** the model in your usual tool ([[Random Forest]], a [[Neural Network]] in [[PyTorch]], and so on).
2. **Export:** a converter turns the trained model into a **graph**: a list of operations (add, multiply, "run this tree ensemble", "apply [[ReLU]]") joined by arrows, plus all the learned numbers (the weights).
3. **Save** it as one `.onnx` file.
4. **Load it with an engine** such as **ONNX Runtime**. The engine reads the graph and runs it on a CPU, a [[GPU]] or a phone, and it can speed things up by fusing and re-ordering steps ([[Kernel Fusion]]).
5. **Check** that the exported model gives the same answers as the original on real data. Never assume.
6. **Watch the versions:** an ONNX file says which "opset" (version of the operator list) it needs. The engine must know that version.

## 🟠 3. Technical detail
- **What is inside.** A graph of operators from named *domains* (the standard one, `ai.onnx`, and `ai.onnx.ml` for classic machine-learning models such as tree ensembles), the weights as tensors, and metadata. The model stores the *computation*, not just the numbers. (Compare [[Safetensors]], which stores only the weights, and [[GGUF]], which stores weights and metadata for [[llama.cpp]].)
- **Numbers are usually 32-bit.** scikit-learn computes in 64-bit; ONNX models commonly use 32-bit, so tiny differences appear (see the result below). A tree that compares a value with a threshold can occasionally send a row down the other branch after rounding.
- **Code: a 20-tree random forest on the diabetes data (442 rows), converted with `skl2onnx`, loaded in ONNX Runtime and compared.** Needs `onnx`, `onnxruntime` and `skl2onnx` (installed in a separate virtual environment, see [[Model Monitoring Lab]]). The converter produces a single operator for the whole forest.

```python
import numpy as np, onnx, onnxruntime as ort
from skl2onnx import to_onnx
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

X, y = load_diabetes(return_X_y=True); X = X.astype(np.float32)           # ONNX models usually work in 32-bit numbers
model = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=0).fit(X, y)

onx = to_onnx(model, X[:1], target_opset={"": 17, "ai.onnx.ml": 3})        # a graph: input -> operators -> output, weights inside
print("operators in the graph:", [n.op_type for n in onx.graph.node])
print("opsets the file asks for:", {o.domain or "ai.onnx": o.version for o in onx.opset_import})
onnx.save(onx, "diabetes_rf.onnx")                                         # one file, no scikit-learn needed to run it

sess = ort.InferenceSession("diabetes_rf.onnx", providers=["CPUExecutionProvider"])   # ONNX Runtime does not import scikit-learn
a = model.predict(X)
b = sess.run(None, {sess.get_inputs()[0].name: X})[0].ravel()
print(f"rows {len(X)} | biggest difference between scikit-learn and ONNX Runtime: {np.abs(a - b).max():.1e}")
print("same ranking of the 442 predictions:", bool((np.argsort(a, kind='stable') == np.argsort(b, kind='stable')).all()))
```

Output (run on my machine, in a fresh folder):

```text
operators in the graph: ['TreeEnsembleRegressor']
opsets the file asks for: {'ai.onnx.ml': 1, 'ai.onnx': 17}
rows 442 | biggest difference between scikit-learn and ONNX Runtime: 3.0e-05
same ranking of the 442 predictions: True
```

- **Reading it:** the whole forest became one `TreeEnsembleRegressor` operator. The two engines agree to $3\times10^{-5}$ in the worst row (the targets are in the hundreds, so for a typical value near 150 the relative error is about $2	imes10^{-7}$, my arithmetic), and the ordering of the 442 predictions is identical. The ONNX file is 75,436 bytes. On the housing model of [[Model Monitoring Lab]] (100 trees, depth 12) the worst difference was $1.7\times10^{-6}$ on 1,136 rows and the file 8.26 MB.
- **Speed (rough).** On the housing model, predicting 1,136 rows took a median of 49.5 ms in scikit-learn against 7.3 ms in ONNX Runtime in one run, and 95.6 ms against 35.1 ms in a second run, so ONNX Runtime was faster both times (about 6.8 times, then about 2.7 times). The laptop was busy and 7 repeats were few, so treat any ratio as rough.
- **What went wrong:** my first export wrote the newest operator-set version (opset 22), and ONNX Runtime 1.19.2 refused to load it: "ONNX Runtime only guarantees support for models stamped with official released onnx opset versions. Opset 22 is under development...". The fix was to ask for an older opset in the converter (`target_opset={"": 17, "ai.onnx.ml": 3}`). This is the most common failure: **the exporter and the engine disagree about versions.** Pin both.

## 🔴 4. Deep / formal
There is no formula, but exactness has limits worth knowing:

- **Parity is a tolerance, not equality.** Check `max |a - b|` on real data and decide what tolerance is acceptable *for the use* (a price in dollars can tolerate $10^{-5}$; a ranking cut-off may not). Check also that the **ranking or decision** is unchanged, as above.
- **32-bit rounding can flip a branch.** A threshold stored in 32 bits may place a value on the other side. Rare, but real in tree models; this is why parity is tested on many rows, not one.
- **Operator coverage.** Not every layer or custom function converts. Unsupported operators fail at export or, worse, are approximated. Test the output.
- **Dynamic shapes and batch size:** the example fixed the number of columns; the number of rows can vary.
- **Opset pinning:** opset numbers grow; an engine supports up to some version. Record the exporter, `onnx` and engine versions together ([[MLOps]]).
- **ONNX is not a security boundary.** A model file can ask for custom operators or heavy computation. Only load files you trust.
- **Other routes to the same goal:** TorchScript, TensorRT, Core ML, TensorFlow Lite and so on (named from memory; not tried). They trade portability against speed on one target.
- **Quantization** (shrinking the numbers to 8-bit or less) is a separate step that ONNX tooling supports (from memory, not tried here); see [[Quantization]].

**What came before, and how this improved on it:**
- **Before:** each framework saved models in its own format and only that framework could run them. Moving a model meant rewriting it. ONNX was announced in 2017 by Facebook and Microsoft and later moved under the Linux Foundation (from memory, not checked).
- **Its limit:** vendor lock-in, heavy dependencies on the serving machine and slow Python-only inference.
- **What changed:** one shared graph format and one fast engine for many sources and targets.
- **How much better:** measured here, on the housing random forest: results within $1.7\times10^{-6}$ of scikit-learn on 1,136 rows, an 8.26 MB file, and ONNX Runtime 2.7 to 6.8 times faster than scikit-learn in two runs on a busy laptop (so the speed-up is only a rough range). Nothing was measured on a neural network, a GPU or a phone. I have no published figure to quote.

## Related
- [[MLOps]] · [[Experiment Tracking]] · [[Data Drift]] · [[Safetensors]] · [[GGUF]] · [[llama.cpp]] · [[Quantization]] · [[Kernel Fusion]] · [[PyTorch]] · [[TensorFlow]] · [[Random Forest]] · [[Neural Network]] · [[GPU]] · [[Docker]] · [[FastAPI]] · [[Model Monitoring Lab]]

## Sources
- My own toy and lab ([[Model Monitoring Lab]]): onnx 1.19.1, onnxruntime 1.19.2, skl2onnx 1.20.0 (installed versions, read with `pip list`). The 2017 announcement, the Linux Foundation move and the other formats are from memory, not checked. Not read: the ONNX specification or documentation.
- No image: I found no free, on-topic explanatory image of ONNX on Wikimedia Commons (only logo files), and did not use an unrelated one.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/mlops ONNX:::Open Neural Network Exchange: a common file format for trained models (a graph of operators plus weights) that many tools can export and many engines, such as ONNX Runtime, can run.

#flashcards/ml/reverse #flashcards/topic/mlops Opset:::The version of the ONNX operator list a model file needs. The engine must support it; mismatched versions are the most common export failure.

#flashcards/ml/concept #flashcards/topic/mlops What is the difference between ONNX, Safetensors and GGUF?
?
ONNX stores the computation (a graph of operators) plus the weights, so an engine can run it. Safetensors stores only the weights safely. GGUF stores weights and metadata for llama.cpp-style engines.

#flashcards/ml/concept #flashcards/topic/mlops What are the steps of using ONNX?
?
Train in your usual tool; export to a graph in an .onnx file; load it in an engine such as ONNX Runtime; check that the answers match the original on real data; watch the opset versions.

#flashcards/ml/concept #flashcards/topic/mlops What did the ONNX toy with the diabetes forest show?
?
The 20-tree forest became one TreeEnsembleRegressor operator. ONNX Runtime and scikit-learn agreed to 3e-05 in the worst of 442 rows (targets in the hundreds) with identical ordering of predictions; the file was 75,436 bytes.

#flashcards/ml/concept #flashcards/topic/mlops What was the opset error in my export, and the fix?
?
ONNX Runtime 1.19.2 refused a model stamped with opset 22 ("Opset 22 is under development"). The fix was asking the converter for an older opset: target_opset={"": 17, "ai.onnx.ml": 3}. The exporter and the engine must agree about versions.

#flashcards/ml/concept #flashcards/topic/mlops Why can an ONNX model differ slightly from the original?
?
ONNX models commonly use 32-bit numbers, while scikit-learn computes in 64-bit. Rounding gives tiny differences, and in tree models a rounded threshold can occasionally send a row down another branch. So test parity on many rows and check that decisions or rankings are unchanged.

#flashcards/ml/concept #flashcards/topic/mlops How fast was ONNX Runtime against scikit-learn on the housing forest, and how reliable is that?
?
1,136 rows: 49.5 ms against 7.3 ms in one run (about 6.8 times) and 95.6 ms against 35.1 ms in a second (about 2.7 times). The laptop was busy and there were only 7 repeats, so only a rough range.

#flashcards/ml/concept #flashcards/topic/mlops Why should you only load ONNX files you trust?
?
A model file can ask for custom operators or heavy computation, so it is not a security boundary.

#flashcards/ml/concept #flashcards/topic/mlops Which ONNX facts are from memory?
?
The 2017 announcement by Facebook and Microsoft, the move to the Linux Foundation, and the other export routes (TorchScript, TensorRT, Core ML, TensorFlow Lite). Not checked.

#flashcards/ml/cloze #flashcards/topic/mlops After an export, always check ==parity==: the ONNX model must give the same answers as the original, within a stated ==tolerance==, on real data.
