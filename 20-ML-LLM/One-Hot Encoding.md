---
tags: [machine-learning, features, preprocessing, categorical-data]
status: growing
level: 4
tier: 1
---

# One-Hot Encoding

## 🟢 1. Simple (age 12)
One-hot encoding means **turning a word-like choice into a row of yes/no switches**. If a pet can be a cat, a dog or a bird, you make three switches: "is it a cat?", "is it a dog?", "is it a bird?". Exactly one switch is on. The computer likes numbers, and this way no animal is "bigger" than another.

**Example:** For a dog the switches read: cat 0, dog 1, bird 0. For a bird: cat 0, dog 0, bird 1.

## 🔵 2. How it works
1. You have a feature with a few named values, such as "chest pain type: A, B, C" or "colour: red, green, blue".
2. The easy idea is to number them: red = 1, green = 2, blue = 3. But that tells the model that blue is "three times" red and that green sits between them, which is nonsense.
3. Instead, make **one new column per value**. Put 1 in the column that matches and 0 in the others.
4. A feature with $n$ values becomes $n$ columns. A feature with only two values (yes/no) needs just one column, so it is left alone.
5. Now a model such as [[Linear Regression]], [[Logistic Regression]], a [[Neural Network]] or a [[Decision Tree]] can use it as ordinary numbers.

## 🟠 3. Technical detail
- **In pandas:** `pd.get_dummies(df, columns=["colour"])` does it ([[Pandas]]). scikit-learn has `OneHotEncoder`. The repo's saved lab turns 5 categorical columns of a heart-disease table into extra columns, taking 11 features to 20 (confirmed by rerunning the lab).
- **Many values = many columns:** a feature with 10,000 values makes 10,000 mostly-zero columns. For such cases people use learned dense vectors instead ([[Embedding]]).
- **Drop one?** The columns always add up to 1, so one is redundant. Plain linear models sometimes drop one column to avoid that; tree models do not need to.
- **Fit on train, apply to test:** the set of values must come from the training data; an unseen value at test time needs a plan (an "unknown" column or ignoring it).
- **Where else it appears:** the target in a classification problem is often one-hot ("class 2 of 4" becomes 0,1,0,0), which is what [[Softmax]] with cross-entropy compares against ([[Loss Function]]).
- **Order is different:** if the values really have an order (small, medium, large), an integer code can be right. One-hot is for values with no order.

```python
import numpy as np
cat = np.array([0, 2, 1, 1])                     # colours as numbers: 0=red, 1=green, 2=blue
print(np.eye(3)[cat])                            # one row of switches per example
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [0. 1. 0.]]
```

## 🔴 4. Deep / formal
**Definition.** A feature that takes one of $n$ values becomes $n$ indicator features. For value $v\in\{1,\dots,n\}$, the encoding is the $v$-th row of the identity matrix. Because exactly one entry is 1, the entries sum to 1.

**Why integers mislead a linear model.** With an integer code $c\in\{0,1,2\}$, a linear model predicts $w c+b$, so the prediction for category 2 is forced to lie "twice as far" from category 0 as category 1 does. With one-hot columns the model has a separate weight per value and can give each category any average it likes.

**Limits:**
- Dimension grows with the number of values, and rare values give columns with almost no data.
- It says nothing about which values are similar (cat and dog vs cat and car). [[Embedding]] vectors can learn that.
- Needs a fixed list of values decided in advance.

**What came before, and how this improved on it:**
- **Before:** replace each category by an integer code.
- **Its limit:** the number invents an order and a distance that do not exist, and a linear model cannot undo it.
- **What changed:** one 0/1 column per category, so each category gets its own weight.
- **How much better:** in the code below (300 examples, three categories whose true averages are 10, 30 and 20, plus small noise), a linear fit on the integer code explained 21.8% of the variation ($R^2=0.218$) and a linear fit on the one-hot columns explained 98.4% ($R^2=0.984$).

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
cat = rng.integers(0, 3, 300)
y = np.array([10., 30., 20.])[cat] + rng.normal(0, 1, 300)     # category means 10, 30, 20
def r2(A, y):
    A = np.c_[A, np.ones(len(y))]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 1 - (y - A @ c).var() / y.var()
print("one-hot R2 %.3f | integer-code R2 %.3f" % (r2(np.eye(3)[cat], y), r2(cat[:, None].astype(float), y)))
```

## Related
- [[Feature Engineering]] · [[Feature Scaling]] · [[Pandas]] · [[Embedding]] · [[Softmax]] · [[Loss Function]] · [[Decision Tree]] · [[Random Forest]] · [[Linear Regression]] · [[Logistic Regression]] · [[Neural Network]] · [[Machine Learning Specialization Course 2 Week 4]] · [[Heart Disease Classifier]]

## Sources
- The repo's saved lab `2. Advanced Algorithms/Week 4/Notebooks/2_Tree_Ensemble.ipynb` (one-hot with `pd.get_dummies`, 11 to 20 features); see [[Machine Learning Specialization]]
- The R² figures are measured in this note's code

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/data-features One-hot encoding:::Turn a named choice into a row of yes/no switches. For cat, dog, bird, make three columns and put 1 in the one that matches and 0 in the others.

#flashcards/ml/example #flashcards/topic/data-features Why not just number the values red = 1, green = 2, blue = 3?::It tells the model that blue is three times red and that green sits between them, which is nonsense for values with no order.

#flashcards/ml/example #flashcards/topic/data-features A feature with $n$ values becomes how many columns? What about a yes/no feature?::$n$ columns. A two-value feature needs just one column, so it is left alone.

#flashcards/ml/example #flashcards/topic/data-features What is the problem with a feature that has 10,000 values?::One-hot makes 10,000 mostly-zero columns. For such cases people use learned dense vectors (embeddings) instead.

#flashcards/ml/example #flashcards/topic/data-features Why do the one-hot columns of one feature always add up to 1, and what does that make redundant?::Exactly one switch is on. So one column is redundant, and plain linear models sometimes drop one.

#flashcards/ml/cloze #flashcards/topic/data-features One-hot encoding is for values with ==no order==. If the values really have an order (small, medium, large), an ==integer code== can be right.

#flashcards/ml/concept #flashcards/topic/data-features Why must the set of values come from the training data?::A test-time value never seen in training needs a plan (an "unknown" column or ignoring it). The columns are fixed by the training data.

#flashcards/ml/concept #flashcards/topic/data-features Where else does one-hot appear in classification?::The target: "class 2 of 4" becomes (0,1,0,0), which is what softmax with cross-entropy compares against.

#flashcards/ml/cloze #flashcards/topic/data-features In pandas, one-hot encoding is done with ==pd.get_dummies==. scikit-learn has `OneHotEncoder`.
