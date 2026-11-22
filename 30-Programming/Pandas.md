---
tags: [pandas, python, programming, data-analysis, tables, machine-learning]
status: growing
level: 4
tier: 2
---

# Pandas

pandas is a [[Python]] library for **working with tables of data**, like a spreadsheet you control with code. You can load a table, clean it, filter it, add columns, summarise it by groups and join it with other tables, all in a few lines. It is built on [[NumPy]] and is the usual first step before [[Machine Learning]].

## 🟢 1. Simple (age 12)
Think of a spreadsheet with rows and columns, like a class register: one row per student, one column for name, one for age, one for marks. pandas lets you give instructions to such a table in writing: "show me only the students over 12", "add a column with marks plus 5", "what is the average mark in each class?". The computer does it for thousands of rows as quickly as for three.

**Example:** A shop writes down every sale: who bought, what, how many, and the price. At the end of the week the owner asks: "How much did each customer spend?" With a pencil, you would go through the list again and again. With pandas you say "group the sales by customer and add up the money", and the answer appears.

## 🔵 2. How it works
1. **Two main things.** A **Series** is one column (a list of values with a label for each). A **DataFrame** is a whole table: several Series side by side that share the same row labels (the **index**).
2. **Load** a table, for example from a CSV file or an Excel sheet. pandas figures out which columns are numbers and which are text.
3. **Look at it first:** the first few rows, how many rows and columns there are, what type each column has, and how many values are missing.
4. **Pick what you need:** choose columns by name, and rows by a condition ("qty is at least 2") or by their labels or positions.
5. **Clean it:** find the missing values (shown as `NaN`, "not a number") and fill or drop them. Fix wrong types, such as numbers stored as text.
6. **Add columns by arithmetic:** `revenue = qty × price` works on the whole column at once, with no loop.
7. **Group and summarise:** split the table by a column (for example by customer), work out something for each group (a sum, an average, a count), and put the answers back into one small table.
8. **Join tables:** attach the city of each customer to the sales table, by matching on the customer name.
9. **Save or hand over:** write the result to a file, or pass the numbers to a [[Machine Learning]] model.

## 🟠 3. Technical detail
- **Index and labels:** every row has a label (by default 0, 1, 2, ...) and every column has a name. `df.loc[rows, cols]` selects by **labels**, and `df.iloc[rows, cols]` selects by **positions**. Mixing them up is a common mistake.
- **Boolean masks:** `df[df['qty'] >= 2]` keeps the rows where the condition is true. Combine conditions with `&` and `|` (with brackets), not `and` and `or`.
- **Vectorised operations:** `df['qty'] * df['price']` runs as one fast array operation (the same idea as in [[NumPy]]). Writing a Python loop over rows, or calling `.apply` with a Python function on each element, is much slower (the code measures it).
- **Missing values:** `NaN` spreads. Any sum that touches a `NaN` becomes `NaN` unless handled (`fillna`, `dropna`, or a function that skips it). Many summaries (`sum`, `mean`) skip `NaN` by default. Check this before trusting a result.
- **`groupby`:** the pattern is *split, apply, combine*. `agg` lets you name several summaries at once. `transform` returns a result as long as the original table (handy for filling a missing value with its group's median, as in the code).
- **`merge` (join):** `how='inner'` keeps only keys found in both tables, `'left'` keeps every row of the left table, `'outer'` keeps all. If a key repeats in both tables, rows **multiply** (equation 2 below). Always check the row count after a join.
- **Index alignment:** adding two Series matches **labels**, not positions. A label found in only one of them gives `NaN` (the code shows it). This is powerful but surprising.
- **Types matter:** `object` (text), `int64`, `float64`, `bool`, `datetime64`, and `category`. Text columns with few distinct values are much smaller as `category` (equation 3 below).
- **Dates:** `pd.to_datetime` turns text into real dates, so you can sort by time, extract the month, or group by week.
- **Copies and views:** assigning to a slice of a table can silently change a copy instead of the original. Use `.loc` to assign, and use `.copy()` when you want a separate table. The rules are changing (a feature called copy-on-write is being introduced), so check the documentation of your version (from memory).
- **Version:** the numbers below were run with pandas 2.2.3, NumPy 1.24.4 and Python 3.9. Output details can differ in other versions.
- **When it is not enough:** pandas holds the whole table in memory. For tables larger than memory, or for very fast work on big tables, other tools exist (from memory: Polars, DuckDB and Spark; I did not test them here).
- **In machine learning:** the usual flow is to load and clean the table with pandas, turn the columns into a [[Matrix]] of numbers, split into training and test parts (see [[Overfitting]] and [[Evaluation Metrics]]), and hand it to a model such as a [[Decision Tree]] or [[Logistic Regression]].

```python
import numpy as np, pandas as pd
orders = pd.DataFrame({
    'order': [1, 2, 3, 4, 5, 6, 7],
    'customer': ['Ann', 'Bob', 'Ann', 'Cy', 'Bob', 'Ann', 'Dee'],
    'product': ['pen', 'pen', 'book', 'book', 'lamp', 'pen', 'lamp'],
    'qty': [2, 1, 1, 3, 1, 4, 2],
    'price': [1.5, 1.5, 12.0, np.nan, 30.0, 1.5, 28.0]})
print(orders.shape, dict(orders.dtypes.astype(str)))
print('missing per column:', orders.isna().sum().to_dict())
# select rows and columns
big = orders[orders['qty'] >= 2]                                   # boolean mask keeps matching rows
print('rows with qty >= 2:', big['order'].tolist(), '| loc:', orders.loc[orders['product'] == 'lamp', ['customer', 'qty']].values.tolist())
# new column (vectorised) and missing values
orders['revenue'] = orders['qty'] * orders['price']                # NaN stays NaN
print('revenue:', orders['revenue'].tolist())
orders['price'] = orders['price'].fillna(orders.groupby('product')['price'].transform('median'))
orders['revenue'] = orders['qty'] * orders['price']
print('after fill, revenue:', orders['revenue'].tolist())
# group and summarise
g = orders.groupby('customer').agg(orders=('order', 'count'), items=('qty', 'sum'), spent=('revenue', 'sum'))
print(g.round(2).to_dict('index'))
print('by product (mean price):', orders.groupby('product')['price'].mean().round(2).to_dict())
# join two tables: a customer who orders twice meets a city table with a repeated key
cities = pd.DataFrame({'customer': ['Ann', 'Ann', 'Bob', 'Cy', 'Eve'], 'city': ['Oslo', 'Rome', 'Paris', 'Lima', 'Kyiv']})
for how in ('inner', 'left', 'outer'):
    m = orders.merge(cities, on='customer', how=how); print(how, 'rows:', len(m))
a = orders['customer'].value_counts(); b = cities['customer'].value_counts()
print('inner rows by formula:', int(sum(a[k] * b[k] for k in a.index if k in b.index)))
# index alignment: adding two Series matches labels, not positions
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c']); s2 = pd.Series([10, 20, 30], index=['b', 'c', 'd'])
print('aligned sum:', (s1 + s2).to_dict())
# memory: text column vs categorical
rng = np.random.default_rng(0); n = 1_000_000
words = pd.Series(rng.choice(['pen', 'book', 'lamp', 'desk'], n))
print('object bytes:', int(words.memory_usage(index=False, deep=True)), '| category bytes:', int(words.astype('category').memory_usage(index=False, deep=True)))
# vectorised vs row-by-row: same answer
x = pd.Series(rng.normal(size=100_000)); print('apply equals vectorised:', np.allclose(x.apply(lambda v: v * 2 + 1), x * 2 + 1))
# split-apply-combine by hand equals groupby
manual = {c: float(orders.loc[orders['customer'] == c, 'revenue'].sum()) for c in orders['customer'].unique()}
print('manual equals groupby:', all(abs(manual[c] - g.loc[c, 'spent']) < 1e-9 for c in manual))
# (7, 5) {'order': 'int64', 'customer': 'object', 'product': 'object', 'qty': 'int64', 'price': 'float64'}
# missing per column: {'order': 0, 'customer': 0, 'product': 0, 'qty': 0, 'price': 1}
# rows with qty >= 2: [1, 4, 6, 7] | loc: [['Bob', 1], ['Dee', 2]]
# revenue: [3.0, 1.5, 12.0, nan, 30.0, 6.0, 56.0]
# after fill, revenue: [3.0, 1.5, 12.0, 36.0, 30.0, 6.0, 56.0]
# {'Ann': {'orders': 3, 'items': 7, 'spent': 21.0}, 'Bob': {'orders': 2, 'items': 2, 'spent': 31.5}, 'Cy': {'orders': 1, 'items': 3, 'spent': 36.0}, 'Dee': {'orders': 1, 'items': 2, 'spent': 56.0}}
# by product (mean price): {'book': 12.0, 'lamp': 29.0, 'pen': 1.5}
# inner rows: 9
# left rows: 10
# outer rows: 11
# inner rows by formula: 9
# aligned sum: {'a': nan, 'b': 12.0, 'c': 23.0, 'd': nan}
# object bytes: 60750421 | category bytes: 1000415
# apply equals vectorised: True
# manual equals groupby: True
```
Reading the output:
- **The table:** 7 orders and 5 columns. `price` has one missing value (order 4, a book). The text columns show up as type `object`.
- **Selecting:** four orders have `qty` of 2 or more (orders 1, 4, 6, 7). The two lamp orders were made by Bob (1 lamp) and Dee (2 lamps).
- **`NaN` spreads:** revenue for order 4 is `nan`, because 3 × `nan` is `nan`. Filling the missing price with the median price of its product (a book costs 12.0 in the other book order) makes it 36.0 (3 × 12.0).
- **Grouping:** Ann has 3 orders, 7 items and spent 21.0. Bob spent 31.5, Cy 36.0 and Dee 56.0.
- **Joins:** the city table lists Ann twice. An inner join gives 9 rows (not 7), a left join 10 and an outer join 11. The formula in equation 2 predicts the 9.
- **Alignment:** adding the two Series gives a value only for labels `b` and `c`, which both have (12 and 23). Labels `a` and `d` become `nan`.
- **Memory:** a million words stored as text take about 60.75 million bytes, and as a category about 1.0 million, around 60 times less.
- **Speed (not in the code, since timing varies):** I also timed a Python function applied to one million numbers with `.apply` against the vectorised `x * 2 + 1`. On my machine, the vectorised version was about 60 times faster (0.44 s against 0.007 s). Your numbers will differ, but the gap is large everywhere.

## 🔴 4. Deep / formal

**Group by, written out.** For a table with a key column, the group summary is

$$y_g=f\big(\{\,x_i : k_i=g\,\}\big)\quad\text{for each distinct key }g$$

**Where:**
- $k_i$ is the key of row $i$ (for example the customer name), and $g$ is one distinct key value. **Matters:** the key decides which rows go together. A typo in a key (such as "ann" against "Ann") creates a separate group.
- $x_i$ is the value in row $i$ that you want to summarise (for example the revenue). **Matters:** only the rows with key $g$ are passed to the summary.
- $f$ is the summary function: a sum, a mean, a count, a median. **Matters:** it is applied to each group separately, so the result has one row per distinct key.
- $y_g$ is the result for group $g$. **Matters:** these results are put back together as a new table (the "combine" step).
- The three steps are *split* (form the groups), *apply* ($f$ to each) and *combine* (stack the $y_g$). **Matters:** this is the pattern behind most reports.

**Example (tiny numbers):** Customer Ann has three orders with revenue $3.0$, $12.0$ and $6.0$. Take $f=\text{sum}$.
1. Split: the three rows with key "Ann" go into one group.
2. Apply: $3.0+12.0+6.0=21.0$.
3. The code's `groupby` gives `spent` $=21.0$ for Ann, and a hand-written loop over the keys gives the same answer.
4. Meaning: the 7-row table is shrunk to 4 rows (one per customer), and each row is an answer to "how much did this customer spend?".

**How many rows a join gives.** Let table A have $a_k$ rows with key $k$, and table B have $b_k$ rows with key $k$. Then

$$N_{\text{inner}}=\sum_{k}a_k\,b_k,\qquad N_{\text{left}}=N_{\text{inner}}+\sum_{k\in A\setminus B}a_k,\qquad N_{\text{outer}}=N_{\text{left}}+\sum_{k\in B\setminus A}b_k$$

**Where:**
- $a_k$ and $b_k$ are how many times key $k$ appears in each table. **Matters:** a key that appears twice on each side gives $2\times2=4$ rows, since every row on one side pairs with every matching row on the other. This is how joins "explode" in size.
- $\sum_k$ adds over every key. **Matters:** only keys present in *both* tables add something to the inner join, because $a_kb_k=0$ if either count is 0.
- $A\setminus B$ means the keys found in A but not in B (read "A without B"). **Matters:** a left join keeps those rows, with empty (`NaN`) values on the B side.
- $B\setminus A$ is the same for keys found only in B. **Matters:** only an outer join keeps these.
- $N$ is the number of rows in the result. **Matters:** compare it with the size you expected. If it is larger than either input, you probably have duplicate keys.

**Example (tiny numbers):** Orders have Ann 3, Bob 2, Cy 1, Dee 1. Cities have Ann 2, Bob 1, Cy 1, Eve 1.
1. Inner: Ann $3\times2=6$, Bob $2\times1=2$, Cy $1\times1=1$. Total $N_{\text{inner}}=9$.
2. Left: add the orders whose customer is missing from the city table. Dee has 1. $N_{\text{left}}=9+1=10$.
3. Outer: add the cities whose customer has no orders. Eve has 1. $N_{\text{outer}}=10+1=11$.
4. The code gives 9, 10 and 11.
5. Meaning: the join of 7 orders and 5 city rows produced 9 rows, because Ann has two cities. Ann's three orders each appear twice, so her sales would be counted twice in any total made after this join. Check the row count after every join.

**What a column costs in memory.** For a column of $n$ values with $k$ distinct texts of average size $\bar s$ bytes:

$$M_{\text{object}}\approx n\,(p+\bar s),\qquad M_{\text{category}}\approx n\,c+k\,\bar s$$

**Where:**
- $n$ is the number of rows. **Matters:** both costs grow with it, but at very different rates.
- $p$ is the size of a pointer (8 bytes on a 64-bit machine). **Matters:** an `object` column stores one pointer per row, each pointing to a Python text object.
- $\bar s$ is the size of one text object (about 50 bytes or more for a short word, because of Python's overhead). **Matters:** in an `object` column, you pay it for each row, even if the same word repeats.
- $c$ is the size of a category code (1 byte when there are fewer than 128 categories). **Matters:** a category column stores only a small number per row and keeps one copy of each distinct text.
- $k$ is the number of distinct values. **Matters:** the category saving is large when $k$ is small compared with $n$, and disappears if nearly every value is different.

**Example (tiny numbers):** $n=1{,}000{,}000$ rows and $k=4$ words (pen, book, lamp, desk) with an average text size of 52.75 bytes (measured from how pandas counted it).
1. Object: $10^6\times(8+52.75)=60{,}750{,}000$ bytes. pandas reported $60{,}750{,}421$.
2. Category: $10^6\times1+4\times52.75=1{,}000{,}211$ bytes. pandas reported $1{,}000{,}415$ (a few hundred bytes of overhead).
3. Ratio: about $60.7$ times smaller.
4. Meaning: storing a repeated label as a small code instead of a full text object cuts memory by a factor of about 60 here. The estimate is close to the measurement, and the exact bytes depend on the pandas and Python versions.

**Other points (from memory, not checked here):**
- pandas was created by Wes McKinney (started around 2008) and is described in his book *Python for Data Analysis*. I did not open either source for this note.
- Its data model borrows ideas from R's data frames and from database tables. A `merge` is a database join.
- Copy-on-write rules and the exact behaviour of chained assignment are still changing between versions, so check the documentation of the version you use.
- I only ran tiny tables and one million-row test. Behaviour on very large tables, many columns or messy files (wrong types, odd dates) is not covered here.

**What came before, and how this improved on it:**
- **Before:** loops over rows, with dictionaries to collect totals and counts.
- **Its limit:** every table question needs a new hand-written loop, and missing values need special cases.
- **What changed:** a table object with grouping, joining and missing-value handling built in. Wes McKinney started pandas in 2008 (from memory, not checked).
- **How much better:** in the code below, the average sales per city over 200,000 rows took a 9-line loop (0.280 s in my first run, 0.089 s in later runs) or one line of pandas (0.0507 s, then about 0.010 s), roughly 6 to 9 times faster, and the answers agreed. The timing is noisy and depends on the machine. With 10% of the values missing, pandas skipped them and still gave 49.96.

**Check it yourself:**

```python
import time, numpy as np, pandas as pd
rng = np.random.default_rng(0); n = 200_000
df = pd.DataFrame({"city": rng.choice(list("ABCDE"), n), "sales": rng.random(n) * 100})
t0 = time.perf_counter(); tot, cnt = {}, {}
for c, s in zip(df["city"], df["sales"]): tot[c] = tot.get(c, 0) + s; cnt[c] = cnt.get(c, 0) + 1
loop = {c: tot[c] / cnt[c] for c in tot}; t1 = time.perf_counter()
g = df.groupby("city")["sales"].mean(); t2 = time.perf_counter()
print("average sales per city, both ways agree:", all(abs(loop[c] - g[c]) < 1e-9 for c in loop))
print(f"  hand-written loop {t1 - t0:.3f} s (9 lines), pandas one line {t2 - t1:.4f} s, ratio about {(t1 - t0) / max(t2 - t1, 1e-9):,.0f}x (depends on this machine)")
df.loc[::10, "sales"] = np.nan; print("with 10% missing values, pandas mean still works:", round(df.groupby("city")["sales"].mean().mean(), 2), "; a plain loop would hit nan and return nan")
```

## Related
[[Python]] · [[NumPy]] · [[Machine Learning]] · [[Matrix]] · [[Supervised Learning]] · [[Evaluation Metrics]] · [[Overfitting]] · [[Decision Tree]] · [[Logistic Regression]] · [[Probability]] · [[Matplotlib]] · [[PyTorch]] · [[Big-O]] · [[Hash Tables]] · [[Binary Search]] · [[Sorting]] · [[Git]] · [[Virtual Environments]] · [[SQL]] · [[Regular Expressions]] · [[One-Hot Encoding]]
- [[Jupyter Notebook]]

## Sources
- pandas documentation, *10 minutes to pandas* and the user guide on `groupby` and `merge` (pandas.pydata.org).
- McKinney, *Python for Data Analysis* (from memory; I did not open it for this note).
- The code, the join and memory counts and every Example number were run for this note (pandas 2.2.3, NumPy 1.24.4, Python 3.9). The shop data is invented for teaching.
- No image: the only on-topic pandas pictures on Wikimedia Commons I found were a screenshot that carries another website's logo (its free licence is doubtful) and a Python 2 code screenshot.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/data-libs pandas:::A spreadsheet you can instruct in writing: "show me only the students over 12", "add a column", "what is the average mark in each class?".

#flashcards/programming/concept #flashcards/topic/data-libs Series against DataFrame?::A Series is one column (values with a label for each). A DataFrame is a whole table of Series side by side that share the same row labels (the index).

#flashcards/programming/concept #flashcards/topic/data-libs What should you do first after loading a table?::Look at the first few rows, the number of rows and columns, each column's type, and how many values are missing.

#flashcards/programming/concept #flashcards/topic/data-libs How are missing values shown, and what do you do?::As `NaN` ("not a number"). Fill or drop them, and fix wrong types such as numbers stored as text.

#flashcards/programming/concept #flashcards/topic/data-libs How do you add a column by arithmetic?::`revenue = qty × price` works on the whole column at once, with no loop.

#flashcards/programming/formula #flashcards/topic/data-libs Write group-and-summarise and explain each symbol.
?
$$y_g=f\big(\{\,x_i : k_i=g\,\}\big)\quad\text{for each distinct key }g$$
- $k_i$: the key of row $i$ (for example the customer name). $g$: one distinct key value. The key decides which rows go together. A typo such as "ann" against "Ann" makes a separate group.
- $x_i$: the value in row $i$ you want to summarise (for example the revenue). Only rows with key $g$ are passed to the summary.
- $f$: the summary function, such as a sum, mean, count or median, applied to each group.
- $y_g$: the answer for group $g$.

#flashcards/programming/example #flashcards/topic/data-libs Ann has three orders with revenue 3.0, 12.0 and 6.0. What is her sum?::The three "Ann" rows form one group. $3.0+12.0+6.0=21.0$ (the code's `groupby` gave the same). The 7-row table shrinks to 4 rows, one per customer.

#flashcards/programming/formula #flashcards/topic/data-libs Write the row counts of inner, left and outer joins and explain each symbol.
?
$$N_{\text{inner}}=\sum_{k}a_k\,b_k,\qquad N_{\text{left}}=N_{\text{inner}}+\sum_{k\in A\setminus B}a_k,\qquad N_{\text{outer}}=N_{\text{left}}+\sum_{k\in B\setminus A}b_k$$
- $a_k$, $b_k$: how many times key $k$ appears in each table. A key that appears twice on each side gives $2\times2=4$ rows, since every row pairs with every matching row, which is how joins explode in size.
- $\sum_k$: adds over every key. Only keys in both tables add to the inner join, because $a_kb_k=0$ if either count is 0.
- $A\setminus B$: keys in A but not in B. Left join adds those rows of A, outer adds the rows of B whose key is missing from A.

#flashcards/programming/example #flashcards/topic/data-libs Orders: Ann 3, Bob 2, Cy 1, Dee 1. Cities: Ann 2, Bob 1, Cy 1, Eve 1. What are the three join sizes?::Inner: $3\cdot2+2\cdot1+1\cdot1=9$. Left: add Dee's 1, so 10. Outer: add Eve's 1, so 11 (the code gives 9, 10, 11).

#flashcards/programming/formula #flashcards/topic/data-libs Write the memory of an `object` column against a `category` column and explain each symbol.
?
$$M_{\text{object}}\approx n\,(p+\bar s),\qquad M_{\text{category}}\approx n\,c+k\,\bar s$$
- $n$: the number of rows. Both costs grow with it, at very different rates.
- $p$: the size of a pointer (8 bytes on a 64-bit machine). An `object` column stores one pointer per row.
- $\bar s$: the size of one text object (about 50 bytes or more for a short word). In an `object` column you pay it for every row, even for repeated words.
- $c$: the size of a small integer code per row. $k$: the number of distinct labels, each stored once.

#flashcards/programming/example #flashcards/topic/data-libs $n=10^6$ rows, $k=4$ words, text size 52.75 bytes (measured). Compare the two columns.::Object: $10^6\times(8+52.75)=60{,}750{,}000$ bytes (pandas reported 60,750,421). Category with 1-byte codes: $10^6+4\times52.75=1{,}000{,}211$ bytes (pandas reported 1,000,415). About 60.7 times smaller. Storing a repeated label as a small code instead of a full text object saves most of the memory.

#flashcards/programming/concept #flashcards/topic/data-libs How do you join a customer's city onto the sales table?::Join (merge) the two tables by matching on the customer name.
