---
tags: [regular-expressions, regex, python, strings, pattern-matching, programming, tooling]
status: growing
level: 4
tier: 2
---

# Regular Expressions

A **regular expression** (regex) is a short code that describes a *pattern of text*, so a program can find, check or rewrite every piece of text that fits the pattern. Instead of asking "is the word `cat` here?", you can ask "is there anything that looks like a phone number here?". Regexes are used in search boxes, code editors, [[Command Line]] tools such as `grep`, and in [[Python]] through the built-in `re` module. They are also handy for cleaning text before it goes into a model (see [[Tokenization]]).

## 🟢 1. Simple (age 12)
Imagine a stamp with holes in it. You press it on a page, and it shows only the bits of text that fit the holes. A regex is a stamp you write yourself out of symbols. One symbol means "any digit", another means "one or more of these", and so on. You slide the stamp along the text, and it lights up everything that fits.

**Example:** You want every phone number in a long message. The pattern `\d{3}-\d{4}` means "three digits, a dash, four digits". It finds `555-0199` and `555-0123` and ignores everything else.

## 🔵 2. How it works
1. **Most characters match themselves.** The pattern `cat` matches the text `cat`. Nothing special.
2. **A few symbols are special.** `.` means "any one character". `\d` means "one digit", `\w` means "one letter, digit or underscore", `\s` means "one space-like character". Square brackets make your own set: `[aeiou]` means "one vowel".
3. **Quantifiers say how many.** `+` means "one or more", `*` means "zero or more", `?` means "maybe once", and `{3}` means "exactly three". So `h[aeiou]+` means "an h followed by one or more vowels".
4. **The engine slides along the text.** It tries the pattern at the first position, then the next, and so on. At each position it works through the pattern left to right and, if something fails, it *backs up* and tries another way (this is called [[Backtracking]]).
5. **Greedy by default.** A quantifier grabs as much text as it can, then gives some back if the rest of the pattern needs it. Add a `?` after it to make it grab as little as possible instead (*lazy*).
6. **Groups and anchors.** Round brackets `( )` make a *group*, which remembers the piece of text it matched so you can use it later. `^` and `$` pin the pattern to the start and the end, and `\b` pins it to the edge of a word.
7. **Under the hood** a regex can be turned into a small machine that reads the text one character at a time, a [[Finite Automaton]]. That is why plain regexes are so fast, and why the fancy extras (like remembering a group and repeating it) are where the trouble starts.

![[regular-expression-example.png]]
*The pattern `h[aeiou]+` (with the `g` flag, "find all") applied to the opening lines of the Bible, with each match shaded: `he` in every "the", `hea` in "heaven", `hou` in "without", `ha` in "that". I counted 17 shaded pieces, and the same pattern run in Python on that text gives 17 as well (14 of them are just `he`). The dots in the picture are the editor's way of showing spaces. Image: Belbury, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Regular_expression_example.png), CC0.*

## 🟠 3. Technical detail
All of this ran in Python 3.9.6 with the built-in `re` module. Other languages and tools have their own versions of regex, with small differences. I only tested Python's.

- **Finding things:** `re.findall(r'\d+', text)` returns every run of digits (`555`, `0199`, `555`, `0123`, `9`, `42`) and `r'\d{3}-\d{4}'` finds the two phone-like pieces. `re.search` finds the first match anywhere, `re.match` only matches at the very start, and `re.fullmatch` demands that the *whole* string fits. The code shows `match` failing on `'a1'` while `search` succeeds.
- **Greedy against lazy:** on `<b>bold</b> and <i>it</i>`, the greedy `<.+>` returned *one* big match covering the whole string, while the lazy `<.+?>` and the tighter `<[^>]+>` each returned the four tags. The tighter one (say what you *don't* want) is usually safer than the lazy one.
- **The `$` trap:** `$` also matches just before a newline at the very end. In the code, `a$` matched `'a\n'`, while `fullmatch` of `a` did not. If you want "the whole string and nothing else", use `fullmatch`.
- **Groups:** `(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})` on `on 2024-03-15 we met` gave the groups `('2024', '03', '15')`, reachable by name (`m.group('y')`) or position, and the match sat at positions 3 to 13. `re.sub` can reuse groups with `\3/\2/\1`, turning `2024-03-15` into `15/03/2024`. A *backreference* (`\1` inside the pattern) matches the same text again, so `\b(\w+) \1\b` found the repeated words `is` and `the` in `it is is the the end`.
- **A regex checks shape, not meaning:** `\d{4}-\d{2}-\d{2}` happily accepted `2024-13-45`, which is not a real date. Use a date library for the meaning, and the regex for a first filter.
- **Raw strings:** write patterns as `r'...'`. A normal Python string turns `'\b'` into one backspace character (length 1), while `r'\b'` stays as two characters, backslash and b (length 2), which `re` reads as "word edge". The code shows `'\bcat\b'` finding nothing and `r'\bcat\b'` finding `cat`.
- **Unicode digits:** in text patterns `\d` matches digits from other scripts too. In the code, `\d` found 2 digits in the string "Arabic-Indic 3, then 4", and only 1 with the `re.ASCII` flag.
- **Overlaps:** `findall('aa', 'aaaa')` found 2 matches, because it moves past each match. A *lookahead* `(?=aa)` matches without using up the text, so it found 3.
- **Escaping:** if the pattern comes from a user or a file, wrap it in `re.escape`. In the code, the raw text `a.b*c` matched the unrelated text `aXbbc` (because `.` and `*` are special), while `re.escape('a.b*c')` did not and matched only the literal text.
- **Flags and readability:** `re.I` ignores case (found `Cat`, `cat`, `CAT`). `re.X` (verbose) lets you spread a pattern over lines with comments, which is far easier to read. `re.split` cuts text at every match.
- **Checking a regex properly:** the code compared the pattern `[01]*1[01]{2}` ("the third character from the end is a 1") with the plain rule `s[-3] == '1'` on *every* 0/1 string of length 0 to 8, which is 511 strings. They agreed on all 511. For small alphabets this is the way to trust a regex: test it against a simple rule on all short inputs.
- **Catastrophic backtracking:** a pattern like `(a+)+b` on a long run of `a` with no `b` at the end can take exponentially long, because the engine tries every way of cutting the `a`s into blocks (see equation 2). I timed one run of each on this computer: `(a+)+b` took 0.004 s at 16 letters, 0.012 s at 18, 0.067 s at 20 and 0.246 s at 22, while the equivalent `a+b` took between 0.02 and 0.08 milliseconds every time. The code itself does not time anything (timings change from run to run), and those four numbers are from one separate run, so treat them as rough. The code also checks that `(a+)+b` and `a+b` agree on all 2,047 strings of `a` and `b` up to length 10, so the safe rewrite changes nothing but the speed.
- **Limits:** I tested only Python's `re`, only short strings, and only the features above. I did not test the third-party `regex` module, lookbehind, conditional patterns, multiline mode, or any other language's engine.

The code covers ten things: (1) classes and quantifiers, (2) greedy against lazy, (3) `match`, `search`, `fullmatch` and the `$` trap, (4) groups and rewriting, (5) raw strings, (6) Unicode digits, (7) an exhaustive check of a pattern, (8) overlaps and the average match count, (9) catastrophic backtracking, (10) escaping, flags and splitting.

```python
import re, itertools
# 1) classes and quantifiers; findall returns every non-overlapping match, left to right
text = 'Call 555-0199 or 555-0123 before 9pm; room 42.'
print('digit runs:', re.findall(r'\d+', text), '| phone-like:', re.findall(r'\d{3}-\d{4}', text), '| words containing p:', re.findall(r'\b\w*p\w*\b', text))
# 2) greedy takes as much as it can, lazy as little as it can
T = '<b>bold</b> and <i>it</i>'
print('greedy <.+> :', re.findall(r'<.+>', T), '| lazy <.+?> :', re.findall(r'<.+?>', T), '| tighter <[^>]+> :', re.findall(r'<[^>]+>', T))
# 3) match / search / fullmatch, and the $ trap
print('match at start:', bool(re.match(r'\d+', 'a1')), '| search anywhere:', bool(re.search(r'\d+', 'a1')), '| fullmatch whole string:', bool(re.fullmatch(r'\d+', '12a')))
print('search a$ in', repr('a\n'), ':', bool(re.search(r'a$', 'a\n')), '| fullmatch a in', repr('a\n'), ':', bool(re.fullmatch(r'a', 'a\n')))
# 4) groups, named groups, rewriting with backreferences
m = re.search(r'(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})', 'on 2024-03-15 we met')
print('groups:', m.groups(), '| named:', m.group('y'), m['m'], m['d'], '| span:', m.span(), '| rewritten:', re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3/\2/\1', '2024-03-15'))
print('repeated words:', re.findall(r'\b(\w+) \1\b', 'it is is the the end'), '| 2024-13-45 fits the date pattern:', bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', '2024-13-45')))
# 5) raw strings: Python reads backslashes first, then re reads what is left
print('len of "\\b":', len('\b'), '| len of r"\\b":', len(r'\b'), '| \\bcat\\b in a normal string finds', re.findall('\bcat\b', 'a cat sat'), '| in a raw string finds', re.findall(r'\bcat\b', 'a cat sat'))
# 6) in text patterns \d also matches digits of other scripts unless told otherwise
s = '٣' + '4'
print('\\d matches in a 2-char string (Arabic-Indic 3, then 4):', len(re.findall(r'\d', s)), '| with re.ASCII:', len(re.findall(r'\d', s, re.ASCII)))
# 7) check a pattern against a plain rule on every 0/1 string up to length 8: "third character from the end is 1"
pat = re.compile(r'[01]*1[01]{2}')
strings = [''.join(p) for n in range(9) for p in itertools.product('01', repeat=n)]
print('regex equals direct rule on', sum(bool(pat.fullmatch(x)) == (len(x) >= 3 and x[-3] == '1') for x in strings), 'of', len(strings), 'strings')
# 8) overlaps: findall skips them, a lookahead counts them; the average count over ALL strings is (n - m + 1) / s^m
print('aa in aaaa: non-overlapping', len(re.findall('aa', 'aaaa')), '| overlapping (lookahead)', len(re.findall('(?=aa)', 'aaaa')))
for n, p in ((6, 'ab'), (6, 'aa'), (10, 'aba')):
    tot = sum(len(re.findall('(?=' + p + ')', ''.join(x))) for x in itertools.product('ab', repeat=n))
    print('n', n, 'pattern', p, '| average count over all', 2 ** n, 'strings =', tot / 2 ** n, '| (n - m + 1) / 2^m =', (n - len(p) + 1) / 2 ** len(p))
# 9) catastrophic backtracking: (a+)+ can cut a run of n a's into blocks in 2^(n-1) ways, and a failing match may try them all
def cuts(n): return 1 if n == 0 else sum(cuts(n - k) for k in range(1, n + 1))
print('ways to cut a^n into blocks, n = 1..8:', [cuts(n) for n in range(1, 9)], '| 2^(n-1):', [2 ** (n - 1) for n in range(1, 9)])
every = [''.join(p) for n in range(11) for p in itertools.product('ab', repeat=n)]
print('(a+)+b and a+b agree on', sum(bool(re.fullmatch(r'(a+)+b', x)) == bool(re.fullmatch(r'a+b', x)) for x in every), 'of', len(every), 'strings | cuts for n = 30:', 2 ** 29)
# 10) escaping user text, flags, a readable verbose pattern, splitting
user = 'a.b*c'
print('unescaped', bool(re.search(user, 'aXbbc')), '| re.escape(user) on that text', bool(re.search(re.escape(user), 'aXbbc')), '| re.escape(user) on the literal text', bool(re.search(re.escape(user), 'xa.b*cx')))
phone = re.compile(r"""
    (\d{3})   # first three digits
    -
    (\d{4})   # last four digits
""", re.X)
print('IGNORECASE:', re.findall(r'cat', 'Cat cat CAT', re.I), '| verbose pattern:', phone.findall(text), '| split:', re.split(r'[,;]\s*', 'a, b;c,d'))
```
Reading the output (Python 3.9.6):
- **Basics:** digit runs `['555', '0199', '555', '0123', '9', '42']`, phone-like `['555-0199', '555-0123']`, and the only word containing a `p` was `9pm`.
- **Greedy:** the greedy pattern gave one match of the whole string. The lazy and the tighter patterns both gave `<b>`, `</b>`, `<i>`, `</i>`.
- **Where it matches:** `match` on `'a1'` was `False` and `search` was `True`. `fullmatch` of digits on `'12a'` was `False`. `a$` found `'a\n'` (`True`) and `fullmatch` did not (`False`).
- **Groups:** `('2024', '03', '15')`, named `2024 03 15`, span `(3, 13)`, rewritten `15/03/2024`, repeated words `['is', 'the']`, and `2024-13-45` fits the date shape (`True`).
- **Raw strings:** length 1 against length 2. The normal string found `[]` and the raw string found `['cat']`.
- **Unicode:** 2 digit matches by default and 1 with `re.ASCII`.
- **Exhaustive check:** the regex equals the direct rule on 511 of 511 strings.
- **Overlaps:** `aa` in `aaaa` gave 2 (non-overlapping) against 3 (lookahead). The three average-count lines gave 1.25, 1.25 and 1.0, each equal to the formula.
- **Backtracking:** the number of cuts for 1 to 8 letters was 1, 2, 4, 8, 16, 32, 64, 128, equal to 2^(n-1). The two patterns agreed on 2047 of 2047 strings, and 30 letters would mean 536,870,912 cuts.
- **Escaping:** the unescaped text `a.b*c` matched `aXbbc` (`True`), the escaped one did not (`False`) and the escaped one found the literal text (`True`). Ignore-case found `Cat`, `cat`, `CAT`, the verbose pattern found `('555', '0199')` and `('555', '0123')`, and `split` gave `['a', 'b', 'c', 'd']`.
- **Limits:** the code runs in a blink, prints and times nothing, and every number comes from tiny inputs.

## 🔴 4. Deep / formal

**How many false alarms?** Suppose you search random text of length $n$ over an alphabet of $s$ equally likely characters for a fixed pattern of $m$ characters. By linearity of expectation, each of the $n-m+1$ possible start positions matches with probability $s^{-m}$, so the expected number of matches (overlaps counted) is:

$$E=\frac{n-m+1}{s^{m}}$$

**Where:**
- $n$ is the length of the text. **Matters:** more text means proportionally more chances for a chance match.
- $m$ is the length of the pattern. **Matters:** each extra character divides the expected count by $s$, so longer patterns are exponentially rarer by chance.
- $s$ is the number of different characters, each equally likely. **Matters:** a bigger alphabet makes every chance match rarer. Real text is far from uniform, so this is a model of random text, not of English.
- $n-m+1$ is the number of places the pattern can start. **Matters:** near the end of the text there is no room left for the pattern to fit.
- $E$ is the expected number of matches when you count overlapping ones too (a lookahead does this; plain `findall` skips overlaps). **Matters:** the formula stays exactly right even if the pattern overlaps itself, as in `aa`, because expectations add up whether or not the events depend on each other.

**Example (tiny numbers):** $n=6$, pattern `ab` ($m=2$), $s=2$ letters (`a` and `b`).
1. There are $n-m+1=5$ start positions.
2. Each matches with probability $2^{-2}=1/4$.
3. $E=5/4=1.25$.
4. The code counted the matches in all $2^6=64$ strings and got 80 in total, and $80/64=1.25$. It gave 1.25 for `aa` as well (it overlaps itself), and 1.0 for `aba` with $n=10$ ($8/8$).
5. Meaning: even a two-letter pattern fires about once per six characters of random text, so a short pattern needs extra context to be trustworthy.

**How slow can backtracking get?** In `(a+)+b` on $n$ letters `a` with no `b`, the engine may try every way to cut the run into consecutive non-empty blocks (a block for each pass of the outer `+`, each block being what the inner `a+` takes). The number of such cuts is:

$$C(n)=2^{\,n-1}$$

**Where:**
- $n$ is the number of `a` characters in the text. **Matters:** each extra letter doubles the count.
- $C(n)$ is the number of ways to cut the run into non-empty blocks (these are called *compositions* of $n$). **Matters:** if the engine tries them all before giving up, the running time grows like $C(n)$ and not like $n$.
- The exponent $n-1$ counts the gaps between neighbouring letters. **Matters:** each of the $n-1$ gaps is either a cut or not, which is two choices per gap and gives $2^{n-1}$ in total.

**Example (tiny numbers):** $n=3$ letters `aaa`.
1. The gaps are between letter 1 and 2, and between 2 and 3, so there are $n-1=2$ gaps.
2. Each gap is cut or not: $2\times2=4$ ways.
3. They are `aaa`, `a|aa`, `aa|a` and `a|a|a`. Counting them by recursion in the code gave 4, and 1, 2, 4, 8, 16, 32, 64, 128 for $n=1$ to $8$, the same as $2^{n-1}$.
4. Meaning: at 30 letters there would be 536,870,912 ways. A real engine does not necessarily try every one (it may stop early or share work), so the timings above grew by roughly three to six times per two extra letters, not exactly four. The shape (growth that keeps multiplying) is what the formula explains. I did not look inside Python's engine to see which paths it actually tries.

**Other points (from memory, not checked here):**
- The word "regular" comes from formal language theory: the patterns built only from characters, "or", grouping and repetition describe exactly the languages that a finite automaton can recognise. Matching those takes time proportional to the length of the text. Features like backreferences (`\1`) go beyond that and cannot be done by a finite automaton, which is why engines that support them use backtracking and can blow up. I did not test this theory here.
- Ken Thompson is usually credited with an early (1960s) construction that turns a regex into an automaton, and it is the reason the Commons picture names `Thompson-...` for the building blocks. I did not check the date or the details.
- Some tools (for example Google's RE2, from memory) refuse backreferences on purpose so that matching time stays linear. I did not test any of them.
- Regexes are famously bad at nested structure such as balanced brackets or full HTML. A real parser is the right tool there, and the `<[^>]+>` example above only works because the tags are simple.
- Python also offers the third-party `regex` module with extra features, and other languages differ in details such as lookbehind and Unicode handling. I did not test them.

**What came before, and how this improved on it:**
- **Before:** write string-scanning code by hand with splits, strips and checks.
- **Its limit:** the code is long, easy to get wrong, and different for every pattern.
- **What changed:** a small language for describing a shape. Kleene described regular events in the 1950s (from memory), and Ken Thompson put them into the text editor QED in the 1960s (from memory, not checked).
- **How much better:** in the code below, finding emails took a hand-written function of 6 lines, and the one pattern `[\w.]+@[\w-]+\.[\w.]+` found the same two emails. A date pattern found both dates. **Catch:** a pattern only checks shape, so the impossible date `2024-13-45` also matches.

**Check it yourself:**

```python
import re
text = "Contact ann@example.com on 2024-03-09 or bob.smith@mail.org by 2025-12-31; ignore a@b and 2024-13."
def by_hand(s):
    out = []
    for w in s.replace(";", " ").replace(",", " ").split():
        w = w.strip(".")
        if "@" in w and "." in w.split("@")[-1] and w.split("@")[0] and w.split("@")[-1][0] != ".": out.append(w)
    return out
pat = r"[\w.]+@[\w-]+\.[\w.]+"
print("emails by a hand-written check:", by_hand(text))
print("emails by one pattern:         ", re.findall(pat, text))
print("dates by one pattern:          ", re.findall(r"\d{4}-\d{2}-\d{2}", text))
print("a pattern is a promise about shape only: 2024-13-45 would also match:", bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", "2024-13-45")))
```

## Related
[[Python]] · [[Command Line]] · [[Tokenization]] · [[Finite Automaton]] · [[Backtracking]] · [[Recursion]] · [[Big-O]] · [[Probability]] · [[Pandas]] · [[SQL]]

## Sources
- Python documentation for the `re` module (docs.python.org), named from memory; I did not open it for this note.
- Python 3.9.6, found by running it. The code and every printed value were run for this note. The four timings in the technical section came from one separate run on this computer and are not repeatable to the digit.
- Figure: [Regular_expression_example.png](https://commons.wikimedia.org/wiki/File:Regular_expression_example.png) (Belbury, CC0), on Wikimedia Commons. I viewed it before writing the caption, and I counted the 17 shaded pieces against Python's output on the same text.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/python-core Regular expression (regex):::A stamp with holes you write yourself out of symbols. You slide it along the text and it lights up everything that fits.

#flashcards/programming/concept #flashcards/topic/python-core What do `.`, `\d`, `\w` and `\s` match, and what do square brackets do?::`.` any one character, `\d` one digit, `\w` one letter, digit or underscore, `\s` one space-like character. Square brackets make your own set: `[aeiou]` is one vowel.

#flashcards/programming/example #flashcards/topic/python-core What do `+`, `*`, `?` and `{3}` mean?::`+` one or more, `*` zero or more, `?` maybe once, `{3}` exactly three. So `h[aeiou]+` is an h followed by one or more vowels.

#flashcards/programming/concept #flashcards/topic/python-core What does greedy mean, and how do you make a quantifier lazy?::A quantifier grabs as much text as it can, then gives some back if the rest of the pattern needs it. Add a `?` after it to grab as little as possible.

#flashcards/programming/example #flashcards/topic/python-core What do parentheses, `^`, `$` and `\b` do?::Round brackets make a group that remembers the text it matched. `^` and `$` pin the pattern to the start and the end of the text. `\b` pins it to the edge of a word.

#flashcards/programming/concept #flashcards/topic/python-core What does the engine do when part of the pattern fails?::It backs up and tries another way (backtracking).

#flashcards/programming/concept #flashcards/topic/python-core Why are plain regexes fast, and where does the trouble start?::A regex can be turned into a small machine that reads the text one character at a time (a finite automaton). Fancy extras, such as remembering a group and repeating it, are where the trouble starts.

#flashcards/programming/formula #flashcards/topic/python-core Write the expected number of chance matches of a pattern in random text and explain each symbol.
?
$$E=\frac{n-m+1}{s^{m}}$$
- $n$: the length of the text. More text means proportionally more chances for a chance match.
- $m$: the length of the pattern. Each extra character divides the expected count by $s$, so longer patterns are exponentially rarer by chance.
- $s$: the number of different characters, each equally likely. A bigger alphabet makes every chance match rarer.
- Real text is far from uniform, so this models random text, not English.

#flashcards/programming/example #flashcards/topic/python-core $n=6$, pattern `ab` ($m=2$), $s=2$ letters. What is $E$?::$n-m+1=5$ start positions, each matching with probability $2^{-2}=1/4$. $E=5/4=1.25$. The note's code counted 80 matches over all 64 strings and $80/64=1.25$. Even a two-letter pattern fires about once per six characters of random text.

#flashcards/programming/formula #flashcards/topic/python-core Write the number of ways to cut a run of letters into blocks and explain each symbol.
?
$$C(n)=2^{\,n-1}$$
- $n$: the number of `a` characters in the text. Each extra letter doubles the count.
- $C(n)$: the number of ways to cut the run into non-empty blocks (compositions of $n$). If the engine tries them all before giving up, running time grows like $C(n)$, not like $n$.
- The exponent $n-1$ counts the gaps between neighbouring letters. Each gap is cut or not, two choices per gap.

#flashcards/programming/example #flashcards/topic/python-core $n=3$ letters `aaa`. How many cuts, and how many at 30 letters?::Gaps $n-1=2$, each cut or not: $2\times2=4$ ways: `aaa`, `a|aa`, `aa|a`, `a|a|a`. The note's code gave 1, 2, 4, 8, 16, 32, 64, 128 for $n=1$ to 8. At 30 letters there would be 536,870,912 ways, though a real engine does not necessarily try every one.

#flashcards/programming/image #flashcards/topic/python-core ![[regular-expression-example.png]]
What does this picture show?
?
The pattern `h[aeiou]+` (with the `g` flag, "find all") applied to the opening lines of the Bible, with each match shaded: `he` in every "the", `hea` in "heaven", `hou` in "without", `ha` in "that". The note counted 17 shaded pieces and the same pattern in Python gave 17 (14 of them just `he`). The dots are the editor's way of showing spaces. (From the note's caption: Belbury, Wikimedia Commons.)
