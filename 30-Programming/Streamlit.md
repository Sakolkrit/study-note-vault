---
tags: [programming, python, web, data-app, tooling]
status: growing
level: 4
tier: 2
---

# Streamlit

**Streamlit** is a Python tool that turns a short script into an interactive web page, with sliders, buttons, tables and charts, without writing any web code.

## 🟢 1. Simple (age 12)
Streamlit lets you build a little web page by writing ordinary Python, almost like writing a list of instructions: "show a title", "show a slider", "show a chart". When someone moves the slider, the page redraws itself with the new answer. You never have to learn how web pages are built.

**Example:** you have a model that guesses whether a student passes. With Streamlit you add one slider for "hours studied". As a friend drags the slider, the page shows the new chance of passing straight away.

## 🔵 2. How it works
1. You write a normal [[Python]] file, for example `app.py`, and `import streamlit as st`.
2. Each line like `st.title(...)`, `st.slider(...)` or `st.line_chart(...)` puts something on the page, in the order you write it.
3. You start it from the [[Command Line]] with `streamlit run app.py`. It opens a page in your browser.
4. A widget such as a slider **returns its current value** as a normal Python variable (`n = st.slider(...)`).
5. When the user touches a widget, Streamlit **runs your whole script again from the top**, with the widget's new value.
6. So the page always matches the code. To remember something between runs, such as a counter, you store it in `st.session_state`.
7. Slow steps (loading a model, reading a big file) would repeat on every run, so you mark them to be **cached**, so they run only once.
8. **The catch:** the whole script reruns on every interaction, so heavy work must be cached, and Streamlit is for quick data apps and demos, not for building a full custom website.

## 🟠 3. Technical detail
- **Execution model:** top-to-bottom script, rerun on every interaction. The code below shows the script running 1, then 2, then 3 times as the slider is moved, and the answer changing each time.
- **State:** `st.session_state` keeps values between reruns for one user's session (the counter in the code below survives the reruns).
- **Caching:** functions that load data or models can be wrapped so the result is reused (the names of the decorators have changed between versions; from memory).
- **Typical use in machine learning:** a demo of a model, an explorer for a data set ([[Pandas]], [[Matplotlib]]), a place to try different settings of an experiment ([[PyTorch]], [[Evaluation Metrics]]).
- **Testing:** newer versions include a tool that runs an app without a browser (used in the code below), so the app can be checked in code ([[pytest]]).
- **With other tools:** a Streamlit page can call a model served by [[FastAPI]], and the app can be shipped in a [[Docker]] container.
- **Compared with a [[Jupyter Notebook]]:** a notebook is for the author to explore, with cells run one at a time. Streamlit is for other people to use, with the code hidden and widgets doing the input.

```python
from streamlit.testing.v1 import AppTest

app = '''
import streamlit as st
st.session_state.runs = st.session_state.get("runs", 0) + 1        # how many times has this script run?
n = st.slider("How many hours did you study?", 0, 10, 2)
st.write(f"chance of passing: {min(0.1 + 0.09 * n, 0.99):.2f}")
st.caption(f"script has run {st.session_state.runs} time(s)")
'''
at = AppTest.from_string(app).run()
print("first visit    :", at.markdown[0].value, "|", at.caption[0].value)
at.slider[0].set_value(8).run()
print("slider moved   :", at.markdown[0].value, "|", at.caption[0].value)
at.slider[0].set_value(3).run()
print("moved again    :", at.markdown[0].value, "|", at.caption[0].value)
print("the whole script re-ran from the top each time; session_state kept the counter between runs")
lines = [l for l in app.strip().splitlines() if l.strip()]
print("the app above is", len(lines), "lines of plain Python, with no HTML, CSS or JavaScript written")
```

## 🔴 4. Deep / formal
**The rerun model.** A Streamlit app is a function of its inputs: the script plus the values of the widgets and the session state. Every interaction produces a new run of the same script, and the page is replaced by what that run draws. This keeps the code simple, because there are no callbacks to wire, and it keeps the page honest, because what you see is exactly what the current values produce.

**Consequences:**
- **Cost:** work that is not cached is repeated on every interaction.
- **State:** anything that must survive a rerun has to be put into `st.session_state` explicitly.
- **Order of widgets:** because the script runs top to bottom, a widget can only influence what is drawn **below** it in the same run.

**Trade-offs:**
- **Very fast to build,** but you give up fine control over layout and behaviour.
- **One script per page** is easy to read, but a large app needs structure.

**What came before, and how this improved on it:**
- **Before:** to show a model to others you built a web server and a front end in HTML and JavaScript (for example a [[FastAPI]] or Flask back end with a hand-written page), or you shared a [[Jupyter Notebook]].
- **Its limit:** the web route needs several languages and many files for a simple demo, and a notebook shows the code to the viewer and is awkward to give to non-programmers.
- **What changed:** widgets are Python function calls and the page reruns the script on each interaction. Streamlit appeared in the late 2010s (from memory, not checked).
- **How much better:** in the code below, the whole app is 5 lines of Python with no HTML, CSS or JavaScript. Moving the slider twice made the script run 3 times in total, and the chance of passing changed from 0.28 (2 hours) to 0.82 (8 hours) to 0.37 (3 hours), while `st.session_state` carried the run counter through. The toy chance formula, $0.1+0.09n$ capped at 0.99, is made up for the demo and is not a trained model.

**Check it yourself:** run the code in level 3 above (it needs `pip install streamlit`, version 1.28 or later for the testing tool; I used 1.50.0). It prints every number quoted in this block.

## Related
- [[Python]] · [[FastAPI]] · [[Docker]] · [[Jupyter Notebook]] · [[Pandas]] · [[Matplotlib]] · [[pytest]] · [[Command Line]] · [[Virtual Environments]] · [[Evaluation Metrics]] · [[Iris Classifier Service]]

## Sources
- Streamlit documentation, https://docs.streamlit.io (the rerun model, `session_state`, caching and the testing tool: from memory, not reopened this session).
- The code and the numbers were run for this note with Streamlit 1.50.0 in a fresh virtual environment.
- No image: I found no free, on-topic picture on Wikimedia Commons, so I added none.
- Names and dates marked "from memory" are not checked.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/web-apps Streamlit:::Build a little web page by writing ordinary Python, almost a list of instructions: "show a title", "show a slider", "show a chart". You never learn how web pages are built.

#flashcards/programming/concept #flashcards/topic/web-apps How do you start a Streamlit app?::From the command line with `streamlit run app.py`. It opens a page in your browser.

#flashcards/programming/concept #flashcards/topic/web-apps How does a widget give you its value?::It returns its current value as a normal Python variable, for example `n = st.slider(...)`.

#flashcards/programming/concept #flashcards/topic/web-apps What happens when the user touches a widget?::Streamlit runs your whole script again from the top, with the widget's new value. So the page always matches the code.

#flashcards/programming/concept #flashcards/topic/web-apps How do you remember something between runs, such as a counter?::Store it in `st.session_state`, which keeps values between reruns for one user's session.

#flashcards/programming/concept #flashcards/topic/web-apps Why cache slow steps?::Loading a model or reading a big file would repeat on every rerun, so you mark them to be cached and they run only once. The decorator names have changed between versions (from memory).

#flashcards/programming/concept #flashcards/topic/web-apps What is the catch of Streamlit?::The whole script reruns on every interaction, so heavy work must be cached. It is for quick data apps and demos, not a full custom website.

#flashcards/programming/concept #flashcards/topic/web-apps Streamlit against a Jupyter notebook?::A notebook is for the author to explore, with cells run one at a time. Streamlit is for other people to use, with the code hidden and widgets doing the input.

#flashcards/programming/concept #flashcards/topic/web-apps How can Streamlit work with FastAPI and Docker?::A Streamlit page can call a model served by FastAPI, and the app can be shipped in a Docker container.
