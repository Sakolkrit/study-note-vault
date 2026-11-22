---
tags: [sql, sqlite, database, programming, data, python]
status: growing
level: 4
tier: 2
---

# SQL

**SQL** (Structured Query Language) is the language for asking questions of a **database**: a place where data is kept in neat tables of rows and columns. Instead of writing loops that go through the data, you describe *what you want* ("the average score for each class"), and the database works out how to get it. Almost every app, website and company keeps its data in a database, so SQL is one of the most useful skills in programming and data work. It sits beside [[Pandas]] in practice: pandas works on tables in memory, and SQL works on tables stored in a database.

## 🟢 1. Simple (age 12)
Picture a school's record book with one page for students (name, class) and another page for exam scores (which student, which subject, what mark). SQL is how you ask the book questions in a very exact way. "Show me the three best maths marks, best first." "For each class, what is the average mark?" "Which students have no marks yet?" You never have to flick through the pages yourself. You write the question, and the book answers.

**Example:** "Give me the names of all students in class A, in alphabetical order" is, in SQL: `SELECT name FROM students WHERE cls = 'A' ORDER BY name`. It reads almost like English.

## 🔵 2. How it works
1. **Data lives in tables.** A *table* has named *columns* (name, class) and many *rows* (one per student). A *primary key* is a column whose values never repeat (like a student number), so each row can be pointed at exactly.
2. **Tables point at each other.** The scores table does not repeat a student's name. It stores the student's number, which matches the number in the students table. This avoids copying the same facts everywhere.
3. **`SELECT` asks a question.** You say which columns you want (`SELECT`), from which table (`FROM`), which rows to keep (`WHERE`), how to order them (`ORDER BY`) and how many to show (`LIMIT`).
4. **`GROUP BY` makes summaries.** It splits the rows into groups, such as one group per class, and gives one answer for each group, such as the average.
5. **`JOIN` combines tables.** A join pairs up rows from two tables whenever a key matches. An *inner* join keeps only the pairs that match. A *left* join also keeps the left table's rows that have no match, filled in with blanks (called `NULL`).
6. **`INSERT`, `UPDATE` and `DELETE` change data.** A *transaction* groups several changes together so that either all happen or none do.
7. **An index makes finding fast.** Without one, the database reads every row to find a match. With an *index* on a column, it can jump straight to the right place, like the index at the back of a book.

![[relational-model.svg]]
*The relational model: data in tables, and tables linked by a shared key. The top-left table lists activity codes (23, 24, 25) and names. The bottom table lists jobs, each with the code of its activity. Following the key "24" from the first table picks out the matching rows of the second (the table on the right). Image: U.S. Department of Transportation (vectorised by a Commons user), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Relational_Model.svg), public domain.*

![[sql-joins.svg]]
*The common kinds of SQL join drawn as overlapping circles (table A on the left, table B on the right, the blue part is what comes out). The inner join keeps only the overlap. A left join keeps all of A, and a right join keeps all of B. A full outer join keeps both. Adding `WHERE B.key IS NULL` (or A) keeps only the rows with no match on the other side. Note that these pictures show *which* rows come out but not *how many*: when keys repeat, one row can pair up with several (see the code and equation 1). Image: Arbeck, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:SQL_Joins.svg), CC BY 3.0.*

## 🟠 3. Technical detail
- **Which database?** There are many (PostgreSQL, MySQL and others, named from memory). This note uses **SQLite**, a small database that comes with Python (`import sqlite3`), needs no setup and stores everything in one file, or in memory as in the code. The code ran SQLite 3.35.5. Other databases speak mostly the same SQL, with differences I did not test.
- **Statements used in the code:** `CREATE TABLE`, `INSERT`, `SELECT ... FROM ... WHERE ... ORDER BY ... LIMIT`, `JOIN`, `LEFT JOIN`, `CROSS JOIN`, `GROUP BY` with `AVG` and `COUNT`, `IS NULL`, `CREATE INDEX`, `EXPLAIN QUERY PLAN`, and the Python calls `commit` and `rollback`.
- **Groups against pandas:** the average score per class from SQL (80.0 for class A and 70.8333 for class B) equals the pandas `groupby(...).mean()` on the same data. Both ignore the one missing score, so class A's average is over 3 marks, not 4. A careless average that counted the blank as 0 would give 60.0 (equation 3 below).
- **Joins:** the code's inner join has 10 rows, the left join 11 (the extra row is Fay, who has no scores), and the cross join (every row paired with every row) has $6\times10=60$. With repeated keys, a join can have *more* rows than either table (equation 1 below).
- **NULL means "unknown":** `score = NULL` finds 0 rows, even though one score is missing, because "unknown equals unknown" is not true. You must write `score IS NULL`. `COUNT(*)` counts all 10 rows, while `COUNT(score)` counts only the 9 that are not `NULL`.
- **Transactions:** the code inserted a student inside a transaction (the count went from 6 to 7) and `rollback` brought it back to 6. A duplicate primary key was refused with `IntegrityError: UNIQUE constraint failed`. I did not test what happens if the program crashes in the middle of a transaction.
- **SQL injection:** building a query by gluing strings together lets the data change the question. With the made-up name `nobody' OR '1'='1`, the glued query returned **2 rows** (all of them), and the same text sent through a `?` placeholder returned 0. **Always pass values as placeholders** (`?` in SQLite, other symbols in other databases). The code uses harmless fake data in memory.
- **Indexes:** the code counted the steps of SQLite's internal machine for one lookup. Without an index the lookup took $3n+10$ steps on a table of $n$ rows (3,010, 30,010 and 300,010 for 1,000, 10,000 and 100,000 rows, equation 2 below). With an index it took 15 steps whatever the size. **The index step is not really free:** a single step hides the work of walking down the index, which I believe grows slowly with $n$ (from memory: indexes are usually a kind of balanced tree). The step count does not show that cost, and I did not time anything. `EXPLAIN QUERY PLAN` says `SCAN TABLE big` without the index and `SEARCH TABLE big USING INDEX ix` with it.
- **Costs of indexes** (from memory; not tested here): they take extra space and make inserts and updates a little slower, because the index must be kept up to date. Index the columns you search or join on often.
- **Not covered or run:** `HAVING`, subqueries, `UNION`, window functions, views, triggers, foreign-key constraints, locking and several users at once, other databases, and query speed on real hardware.
- **Limits:** one in-memory database, tiny tables, and step counts rather than times. The step counts are for SQLite 3.35.5 and may differ in other versions.

The code covers seven things: (1) tables, rows and a first query, (2) `GROUP BY` against pandas, (3) the kinds of join and their row counts, (4) `NULL` and the average, (5) transactions and a primary-key error, (6) injection against placeholders, (7) index against full scan, counted in steps.

```python
import sqlite3
import pandas as pd
con = sqlite3.connect(':memory:'); q = lambda sql, *a: con.execute(sql, a).fetchall()
print('SQLite version:', sqlite3.sqlite_version)
# 1) tables, rows and columns; a primary key is a column whose values never repeat
q('CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, cls TEXT)'); q('CREATE TABLE scores (student_id INTEGER, subject TEXT, score REAL)')
con.executemany('INSERT INTO students VALUES (?,?,?)', [(1, 'Ana', 'A'), (2, 'Ben', 'A'), (3, 'Cy', 'B'), (4, 'Di', 'B'), (5, 'Eli', 'B'), (6, 'Fay', 'A')])
con.executemany('INSERT INTO scores VALUES (?,?,?)', [(1, 'math', 90), (1, 'sci', 80), (2, 'math', 70), (2, 'sci', None), (3, 'math', 60), (3, 'sci', 100), (4, 'math', 85), (4, 'sci', 75), (5, 'math', 50), (5, 'sci', 55)])
print('rows: students', q('SELECT COUNT(*) FROM students')[0][0], '| scores', q('SELECT COUNT(*) FROM scores')[0][0])
print('WHERE + ORDER BY + LIMIT (top 3 math scores):', q("SELECT name, score FROM students JOIN scores ON id = student_id WHERE subject = 'math' ORDER BY score DESC LIMIT 3"))
# 2) GROUP BY: one answer per group; same as pandas groupby
print('average score per class (SQL):', q('SELECT cls, ROUND(AVG(score), 4) FROM students JOIN scores ON id = student_id GROUP BY cls ORDER BY cls'))
df = pd.read_sql('SELECT cls, score FROM students JOIN scores ON id = student_id', con)
print('same with pandas groupby:', [(c, round(m, 4)) for c, m in df.groupby('cls')['score'].mean().items()], '| naive average if the missing score counted as 0 (class A):', round(sum(0 if pd.isna(x) else x for x in df[df.cls == 'A'].score.tolist()) / len(df[df.cls == 'A']), 4))
# 3) joins: INNER keeps matches only, LEFT keeps every row on the left; repeated keys multiply
print('INNER JOIN rows:', q('SELECT COUNT(*) FROM students JOIN scores ON id = student_id')[0][0], '| LEFT JOIN rows:', q('SELECT COUNT(*) FROM students LEFT JOIN scores ON id = student_id')[0][0], '| students with no scores:', q('SELECT name FROM students LEFT JOIN scores ON id = student_id WHERE student_id IS NULL'))
print('CROSS JOIN rows:', q('SELECT COUNT(*) FROM students CROSS JOIN scores')[0][0], '= 6 x 10')
q('CREATE TABLE a (k INTEGER, x TEXT)'); q('CREATE TABLE b (k INTEGER, y TEXT)')
con.executemany('INSERT INTO a VALUES (?,?)', [(1, 'a1'), (1, 'a2'), (2, 'a3'), (3, 'a4')]); con.executemany('INSERT INTO b VALUES (?,?)', [(1, 'b1'), (1, 'b2'), (1, 'b3'), (2, 'b4'), (4, 'b5')])
print('join with repeated keys: rows =', q('SELECT COUNT(*) FROM a JOIN b ON a.k = b.k')[0][0], '| key 1: 2 x 3, key 2: 1 x 1, keys 3 and 4 match nothing')
# 4) NULL means "unknown": comparisons with it are never true
print('score = NULL finds', len(q('SELECT * FROM scores WHERE score = NULL')), 'rows | score IS NULL finds', len(q('SELECT * FROM scores WHERE score IS NULL')), '| COUNT(*) =', q('SELECT COUNT(*) FROM scores')[0][0], 'but COUNT(score) =', q('SELECT COUNT(score) FROM scores')[0][0])
q('CREATE TABLE t (v REAL)'); con.executemany('INSERT INTO t VALUES (?)', [(10,), (20,), (None,)])
print('AVG of 10, 20, NULL =', q('SELECT AVG(v) FROM t')[0][0], '= (10 + 20) / COUNT(v) =', (10 + 20) / q('SELECT COUNT(v) FROM t')[0][0], '| not (10 + 20 + 0) / 3 =', 30 / 3)
# 5) transactions: all or nothing
con.commit(); n0 = q('SELECT COUNT(*) FROM students')[0][0]
q("INSERT INTO students VALUES (7, 'Gus', 'B')"); inside = q('SELECT COUNT(*) FROM students')[0][0]; con.rollback()
print('students before', n0, '| inside the transaction', inside, '| after rollback', q('SELECT COUNT(*) FROM students')[0][0])
try: q("INSERT INTO students VALUES (1, 'Dup', 'A')")
except sqlite3.IntegrityError as e: print('duplicate primary key ->', type(e).__name__ + ':', e)
# 6) injection: building SQL by gluing strings lets the data change the question; placeholders do not
q('CREATE TABLE users (name TEXT, secret TEXT)'); con.executemany('INSERT INTO users VALUES (?,?)', [('amy', 'red'), ('bo', 'blue')])
evil = "nobody' OR '1'='1"
print('glued string finds', len(q("SELECT secret FROM users WHERE name = '" + evil + "'")), 'rows | with a ? placeholder finds', len(q('SELECT secret FROM users WHERE name = ?', evil)), 'rows')
# 7) an index lets the database jump instead of scanning: count the steps of its internal machine, not seconds
def steps(sql, arg):
    c = [0]
    def tick(): c[0] += 1; return 0
    con.set_progress_handler(tick, 1); con.execute(sql, (arg,)).fetchall(); con.set_progress_handler(None, 1); return c[0]
for n in (1000, 10000, 100000):
    q('DROP TABLE IF EXISTS big'); q('CREATE TABLE big (plain INTEGER, indexed INTEGER)')
    con.executemany('INSERT INTO big VALUES (?,?)', ((i, i) for i in range(n))); q('CREATE INDEX ix ON big(indexed)')
    s1, s2 = steps('SELECT * FROM big WHERE plain = ?', n - 1), steps('SELECT * FROM big WHERE indexed = ?', n - 1)
    print('n', n, '| steps, no index:', s1, '| 3n + 10 =', 3 * n + 10, '| steps, with index:', s2)
plan = lambda sql: q('EXPLAIN QUERY PLAN ' + sql, 5)[0][3]
print('query plans:', plan('SELECT * FROM big WHERE plain = ?'), '|', plan('SELECT * FROM big WHERE indexed = ?'))
# SQLite version: 3.35.5
# rows: students 6 | scores 10
# WHERE + ORDER BY + LIMIT (top 3 math scores): [('Ana', 90.0), ('Di', 85.0), ('Ben', 70.0)]
# average score per class (SQL): [('A', 80.0), ('B', 70.8333)]
# same with pandas groupby: [('A', 80.0), ('B', 70.8333)] | naive average if the missing score counted as 0 (class A): 60.0
# INNER JOIN rows: 10 | LEFT JOIN rows: 11 | students with no scores: [('Fay',)]
# CROSS JOIN rows: 60 = 6 x 10
# join with repeated keys: rows = 7 | key 1: 2 x 3, key 2: 1 x 1, keys 3 and 4 match nothing
# score = NULL finds 0 rows | score IS NULL finds 1 | COUNT(*) = 10 but COUNT(score) = 9
# AVG of 10, 20, NULL = 15.0 = (10 + 20) / COUNT(v) = 15.0 | not (10 + 20 + 0) / 3 = 10.0
# students before 6 | inside the transaction 7 | after rollback 6
# duplicate primary key -> IntegrityError: UNIQUE constraint failed: students.id
# glued string finds 2 rows | with a ? placeholder finds 0 rows
# n 1000 | steps, no index: 3010 | 3n + 10 = 3010 | steps, with index: 15
# n 10000 | steps, no index: 30010 | 3n + 10 = 30010 | steps, with index: 15
# n 100000 | steps, no index: 300010 | 3n + 10 = 300010 | steps, with index: 15
# query plans: SCAN TABLE big | SEARCH TABLE big USING INDEX ix (indexed=?)
```
Reading the output (SQLite 3.35.5, pandas 2.2.3, Python 3.9):
- **Rows and a query:** 6 students and 10 scores. The three best maths marks are Ana 90, Di 85 and Ben 70.
- **Averages:** class A 80.0 and class B 70.8333, from SQL and from pandas alike. Counting the blank as 0 would give 60.0 for class A.
- **Joins:** inner join 10 rows, left join 11 (Fay has no scores), cross join 60. With repeated keys the join of tables `a` and `b` has 7 rows: key 1 pairs 2 rows with 3 rows (6), key 2 pairs 1 with 1 (1), and keys 3 and 4 have no match.
- **NULL:** `= NULL` finds 0 rows and `IS NULL` finds 1. `COUNT(*)` is 10 and `COUNT(score)` is 9. The average of 10, 20 and `NULL` is 15.0, which is $(10+20)/2$, not $(10+20+0)/3=10$.
- **Transactions:** 6 students before, 7 inside, 6 after the rollback. A duplicate id raised `IntegrityError: UNIQUE constraint failed: students.id`.
- **Injection:** the glued query found 2 rows and the placeholder query found 0.
- **Index:** without an index 3,010, 30,010 and 300,010 steps (exactly $3n+10$ each time), with an index 15 steps each time. The plans are `SCAN TABLE big` and `SEARCH TABLE big USING INDEX ix (indexed=?)`.
- **Limits:** tiny in-memory tables, counted steps, nothing timed.

## 🔴 4. Deep / formal

**How many rows does a join produce?** Join table $A$ to table $B$ on a key. Let $a_k$ be the number of rows of $A$ whose key is $k$, and $b_k$ the number of rows of $B$ with key $k$. Every row of $A$ with key $k$ pairs with every row of $B$ with key $k$, so an inner join has:

$$|A\bowtie B|=\sum_{k}a_k\,b_k$$

**Where:**
- $k$ stands for each key value that appears. **Matters:** only keys present in *both* tables contribute, because a key missing from one table has $a_k=0$ or $b_k=0$, so its product is 0.
- $a_k$ and $b_k$ are the numbers of rows with key $k$ in each table. **Matters:** if a key is unique on both sides (a primary key joined to a column that points at it once), then every product is 0 or 1 and the join is no bigger than the smaller table. If keys repeat on both sides, the rows multiply.
- $a_k\,b_k$ is the number of pairs for key $k$. **Matters:** this is where surprises come from: 2 rows against 3 rows gives 6, not 3.
- A join with no condition (a cross join) is the special case where every row has the same key: it gives $|A|\times|B|$ rows. **Matters:** it is almost always a mistake when it happens by accident, because the result can be huge.

**Example (tiny numbers):** table `a` has keys 1, 1, 2, 3 and table `b` has keys 1, 1, 1, 2, 4.
1. Key 1: $a_1=2$ and $b_1=3$, so $2\times3=6$ pairs.
2. Key 2: $a_2=1$ and $b_2=1$, so 1 pair. Key 3: $b_3=0$, so 0 pairs. Key 4: $a_4=0$, so 0 pairs.
3. Total: $6+1+0+0=7$. The code's join returned 7 rows.
4. Meaning: even though the two tables have only 4 and 5 rows, the join has 7, more than either. For the cross join of 6 students and 10 scores the formula with one shared key gives $6\times10=60$, which is also what the code returned.

**How much work is a lookup?** To find the rows where a column equals a value, a database without an index must check every row. I measured the number of internal steps for one such query on tables of $n$ rows and found, for these three sizes (exactly, not as an estimate):

$$\text{steps}_{\text{scan}}(n)=3n+10,\qquad \text{steps}_{\text{index}}(n)=15$$

**Where:**
- $n$ is the number of rows in the table. **Matters:** the scan cost grows in proportion to $n$, so a table 100 times bigger takes about 100 times more steps.
- $3n$ is about three internal steps per row (read the row, compare, move on). **Matters:** this "3" and the "+10" for setting up and finishing come from my measurement on this query and this SQLite version. They are not a law, and another query or version can give other numbers.
- $15$ is the steps used by the indexed lookup. **Matters:** it stayed the same for all three sizes, because the descent through the index is done inside one internal step. So this count hides the real, slowly growing cost of the descent (I believe it is about $\log n$ page visits, from memory, but I did not measure it).
- Both formulas count steps, not seconds. **Matters:** steps are a fair measure of work on the same machine, but I did not check how steps turn into time.

**Example (tiny numbers):** a lookup of the last row.
1. $n=1{,}000$: scan $3\times1000+10=3010$ steps. The code counted 3,010. With the index: 15.
2. $n=100{,}000$: scan $3\times100000+10=300{,}010$ steps. The code counted 300,010. With the index: 15.
3. Ratio at $n=100{,}000$: $300{,}010/15\approx20{,}000$ times fewer steps.
4. Meaning: the scan needs 100 times more steps when the table is 100 times bigger, and the indexed lookup does not change in step count at all. This is why a database with millions of rows can still answer a lookup at once, but only if the column you search has an index.

**Averages and the blanks.** SQL's `AVG` ignores `NULL` values. With values $x_1,\dots,x_m$ of which $c$ are not `NULL`, the average is:

$$\operatorname{AVG}=\frac{\sum_{i:\,x_i\ne\text{NULL}}x_i}{c}$$

**Where:**
- $x_i$ are the values in the column, and $m$ is how many rows there are. **Matters:** the rows with `NULL` still exist, but they do not take part in the sum.
- $c$ is the count of the values that are not `NULL` (what `COUNT(column)` returns, not `COUNT(*)`). **Matters:** the divisor is $c$, not $m$. Dividing by $m$ would treat each unknown as a zero and pull the average down.
- `NULL` means "unknown", not zero. **Matters:** an unknown mark is not the same as a mark of 0, so leaving it out is usually the right choice, but you must be aware that it happens. If you want blanks to count as 0, say so explicitly (for example with `COALESCE(score, 0)`, from memory; I did not test it).
- If every value is `NULL`, then $c=0$ and the result is `NULL` (from memory; I did not test it).

**Example (tiny numbers):** values 10, 20 and `NULL`.
1. Sum of the known values: $10+20=30$. Count of known values: $c=2$.
2. $\operatorname{AVG}=30/2=15$. The code returned 15.0.
3. Counting the blank as 0 would give $30/3=10$, which is what a careless calculation produces.
4. Meaning: the database reports the average of what it knows, 15. In the class example, class A's average is 80 over 3 known marks, and treating Ben's missing mark as 0 would wrongly give 60.

**Other points (from memory, not checked here):**
- SQL's logical order of work is: pick the tables and joins (`FROM`), filter rows (`WHERE`), group (`GROUP BY`), filter groups (`HAVING`), choose columns (`SELECT`), sort (`ORDER BY`), cut (`LIMIT`). This is why you cannot use a column's new name from `SELECT` inside `WHERE`. I did not test this order.
- The relational model behind SQL was proposed by Edgar Codd in 1970, and SQL was developed at IBM in the 1970s. I did not check either date.
- Databases have a *query planner* that picks how to run your query. `EXPLAIN QUERY PLAN` showed its choice (scan or index) for the code's two cases. I did not test how it chooses in general.
- A relational database is not the only kind. Others store documents, key-value pairs or graphs. I did not cover them.
- Normalising (splitting data across tables to avoid repeating it) has formal rules called normal forms. I did not cover them.

**What came before, and how this improved on it:**
- **Before:** write a program that loops over the records and says how to find the answer.
- **Its limit:** each question is a new program, and the program has to know how the data is stored.
- **What changed:** say what you want and let the database choose how. Codd proposed the relational model in 1970 and SQL appeared in the 1970s at IBM (from memory, not checked).
- **How much better:** in the code below, one `group by` line gave the same average per city as a Python loop over 100,000 rows. And adding an index changed the plan the database chose from `SCAN TABLE sales` (look at every row) to `SEARCH TABLE sales USING INDEX i (id=?)` (jump straight to it), without changing the question.

**Check it yourself:**

```python
import sqlite3, random
random.seed(0)
db = sqlite3.connect(":memory:"); db.execute("create table sales(id integer, city text, amount real)")
db.executemany("insert into sales values(?,?,?)", [(i, random.choice("ABCDE"), random.random() * 100) for i in range(100000)])
rows = db.execute("select city, round(avg(amount), 4) from sales group by city order by city").fetchall()
tot, cnt = {}, {}
for _, c, a in db.execute("select * from sales"): tot[c] = tot.get(c, 0) + a; cnt[c] = cnt.get(c, 0) + 1
print("average per city, SQL one line vs a Python loop agree:", all(abs(a - tot[c] / cnt[c]) < 1e-3 for c, a in rows))
q = "select * from sales where id = 77777"
print("plan without an index:", db.execute("explain query plan " + q).fetchall()[0][3])
db.execute("create index i on sales(id)")
print("plan with an index:   ", db.execute("explain query plan " + q).fetchall()[0][3])
```

## Related
[[Pandas]] · [[Python]] · [[NumPy]] · [[Hash Tables]] · [[Trees]] · [[Big-O]] · [[Binary Search]] · [[Git]] · [[Vector Database]]

## Sources
- SQLite documentation (sqlite.org), on the language, `EXPLAIN QUERY PLAN` and `NULL` handling (from memory; I did not open the pages for this note).
- Python documentation for the `sqlite3` module (docs.python.org).
- Codd, E. F., "A relational model of data for large shared data banks", 1970 (named from memory; I did not open it).
- Figures: [Relational_Model.svg](https://commons.wikimedia.org/wiki/File:Relational_Model.svg) (U.S. Department of Transportation, public domain) and [SQL_Joins.svg](https://commons.wikimedia.org/wiki/File:SQL_Joins.svg) (Arbeck, CC BY 3.0), on Wikimedia Commons. I viewed both before writing the captions.
- The code and every Example number were run for this note (SQLite 3.35.5, pandas 2.2.3, Python 3.9, in-memory database). Steps were counted, nothing was timed.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/data-libs SQL:::A very exact way to ask a record book questions, such as "the three best maths marks, best first" or "the average mark for each class".

#flashcards/programming/concept #flashcards/topic/data-libs Table, column, row, primary key?::A table has named columns and many rows (one per student). A primary key is a column whose values never repeat, so each row can be pointed at exactly.

#flashcards/programming/concept #flashcards/topic/data-libs Why do tables store a student's number instead of the name in the scores table?::To avoid copying the same facts everywhere. The number matches the key in the students table.

#flashcards/programming/concept #flashcards/topic/data-libs What do `SELECT`, `FROM`, `WHERE`, `ORDER BY` and `LIMIT` do?::`SELECT` picks the columns, `FROM` the table, `WHERE` which rows to keep, `ORDER BY` the order and `LIMIT` how many to show.

#flashcards/programming/concept #flashcards/topic/data-libs What does `GROUP BY` do?::It splits the rows into groups, such as one group per class, and gives one answer for each group, such as the average.

#flashcards/programming/concept #flashcards/topic/data-libs Inner join against left join?::An inner join keeps only the pairs where a key matches. A left join also keeps the left table's rows with no match, filled with blanks (`NULL`).

#flashcards/programming/concept #flashcards/topic/data-libs What do `INSERT`, `UPDATE` and `DELETE` do, and what is a transaction?::They change data. A transaction groups several changes so that either all happen or none do.

#flashcards/programming/concept #flashcards/topic/data-libs What is an index?::A structure on a column that lets the database jump straight to the right place instead of reading every row, like the index at the back of a book.

#flashcards/programming/formula #flashcards/topic/data-libs Write the number of rows in a join and explain each symbol.
?
$$|A\bowtie B|=\sum_{k}a_k\,b_k$$
- $k$: each key value that appears. Only keys present in both tables contribute, because a missing key has $a_k=0$ or $b_k=0$.
- $a_k$, $b_k$: the numbers of rows with key $k$ in each table.
- If a key is unique on both sides (a primary key joined to a column pointing at it once) every product is 0 or 1 and the join is no bigger than the smaller table. If keys repeat on both sides, the join can be bigger than either table.

#flashcards/programming/example #flashcards/topic/data-libs Table `a` has keys 1, 1, 2, 3 and table `b` has keys 1, 1, 1, 2, 4. How many rows does the join return?::Key 1: $2\times3=6$. Key 2: $1\times1=1$. Key 3: $b_3=0$. Key 4: $a_4=0$. Total $6+1=7$ (the code returned 7), more than either table's 4 or 5 rows.

#flashcards/programming/formula #flashcards/topic/data-libs Write the steps of a table scan and of an index lookup and explain each symbol.
?
$$\text{steps}_{\text{scan}}(n)=3n+10,\qquad \text{steps}_{\text{index}}(n)=15$$
- $n$: the number of rows in the table. The scan cost grows in proportion to $n$, so a table 100 times bigger takes about 100 times more steps.
- $3n$: about three internal steps per row (read the row, compare, move on). The 3 and the +10 come from the note's own measurement on this query and this SQLite version, not a law.
- 15: the steps used by the index lookup, which did not change with $n$.

#flashcards/programming/example #flashcards/topic/data-libs Lookup of the last row at $n=1{,}000$ and $n=100{,}000$. What do the scan and the index take?::$n=1{,}000$: scan $3010$ steps (the code counted 3,010), index 15. $n=100{,}000$: scan $300{,}010$ (counted 300,010), index 15. Ratio about 20,000 times fewer. The scan needs 100 times more steps for a 100 times bigger table, while the index lookup stays the same.

#flashcards/programming/formula #flashcards/topic/data-libs Write the SQL average and explain each symbol.
?
$$\operatorname{AVG}=\frac{\sum_{i:\,x_i\ne\text{NULL}}x_i}{c}$$
- $x_i$: the values in the column. Rows with `NULL` still exist but do not take part in the sum.
- $c$: the count of values that are not `NULL` (what `COUNT(column)` returns, not `COUNT(*)`). The divisor is $c$, not the number of rows. Dividing by the row count would treat each unknown as a zero and pull the average down.
- `NULL` means unknown, not zero.

#flashcards/programming/example #flashcards/topic/data-libs Values 10, 20 and `NULL`. What is `AVG`?::Sum of known values $30$, count of known values $c=2$, so $15$ (the code returned 15.0). Counting the blank as 0 would give $30/3=10$. In the note's class example, class A's average is 80 over 3 known marks, and treating a missing mark as 0 would wrongly give 60.

#flashcards/programming/image #flashcards/topic/data-libs ![[relational-model.svg]]
What does this picture show?
?
The relational model: data in tables, linked by a shared key. The top-left table lists activity codes (23, 24, 25) and names. The bottom table lists jobs, each with the code of its activity. Following the key "24" from the first table picks out the matching rows of the second (the table on the right). (From the note's caption: U.S. Department of Transportation, vectorised by a Commons user, Wikimedia Commons.)

#flashcards/programming/image #flashcards/topic/data-libs ![[sql-joins.svg]]
What does this picture show?
?
The common kinds of SQL join as overlapping circles (table A left, table B right, the blue part is what comes out). The inner join keeps only the overlap. A left join keeps all of A and a right join keeps all of B. A full outer join keeps both. Adding `WHERE B.key IS NULL` (or A) keeps only the rows with no match on the other side. The pictures show which rows come out, not how many. (From the note's caption, Wikimedia Commons.)

#flashcards/programming/cloze #flashcards/topic/data-libs `NULL` means ==unknown==, not zero.
