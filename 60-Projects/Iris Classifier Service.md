---
tags: [project, python, FastAPI, Streamlit, Docker, machine-learning, deployment]
status: done
stack: Python 3.9, scikit-learn, FastAPI, Streamlit, Docker
goal: train a tiny model, serve it behind an API, give it a web page, and package both
---

# Iris Classifier Service

> **In one line:** train a small flower-species model, serve it with [[FastAPI]], put a [[Streamlit]] page in front of it, and package the two in [[Docker]]: the whole path from "a model in a file" to "a service anyone can call".

## Goal
Done means four checks pass: (1) the model trains and is saved, (2) the API answers a request with a species and chances, and refuses bad input, (3) the web page shows a prediction as the sliders move, (4) the two services are described so that `docker compose up` could start them.

## Result in short
- The model is a [[Logistic Regression]] on the 150-flower iris data set. It scored **1.0 accuracy on 45 held-out flowers** (a toy data set, so this says little; see [[Evaluation Metrics]]).
- The API answered a setosa request with `setosa` and chances 0.976 / 0.024 / 0.0, in **about 25 ms** for one local call on my machine (one call, not a benchmark). A negative measurement was refused with status 422.
- 5 tests pass with [[pytest]].
- The Streamlit page, driven by Streamlit's own test tool against the real running server, showed `versicolor` for the default sliders, then `setosa` and `virginica` after I moved the sliders to typical values for those species. With the API stopped, it showed a clear error instead of crashing.
- **Docker:** both images built from the Dockerfiles in this note and `docker compose up` started both containers (API image 577 MB, page image 751 MB, as Docker reported them). Through the containers: `/health` answered ok, the setosa request returned the same `setosa` 0.976 / 0.024 / 0.0 (one call, 33 ms), a negative value gave 422, and the page at `http://localhost:8501` answered 200. From inside the page's container, a call to `http://api:8000/predict` returned `virginica` for 6.9, 3.1, 5.4, 2.1, so the container-to-container address works. A first build took well under a minute on my machine (one run, not a benchmark).

## What you need
Versions I used (a fresh virtual environment, Windows, Python 3.9):

| Tool | Version |
|---|---|
| scikit-learn | 1.6.1 |
| joblib | 1.5.3 |
| FastAPI | 0.128.8 |
| uvicorn | 0.39.0 |
| Streamlit | 1.50.0 |
| httpx | 0.28.1 |
| pytest | 8.4.2 |
| Docker | 28.0.4 (Docker Desktop, Linux containers), base image `python:3.9-slim` |

## Steps
1. **Make a folder and a clean environment** ([[Virtual Environments]], [[Command Line]]). In the folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install scikit-learn==1.6.1 joblib==1.5.3 fastapi==0.128.8 uvicorn==0.39.0 streamlit==1.50.0 httpx==0.28.1 pytest==8.4.2
   ```
   (On Mac or Linux, activate with `source .venv/bin/activate`.)

2. **Train the model.** This is plain [[Supervised Learning]]: split the data, fit, check on the held-out part, save to a file.

   **`train.py`**
   ```python
   import joblib
   from sklearn.datasets import load_iris
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import train_test_split

   X, y = load_iris(return_X_y=True)
   Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
   model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
   print("test accuracy:", round(model.score(Xte, yte), 3))
   joblib.dump(model, "model.joblib")
   print("saved model.joblib")
   ```
   Run `python train.py`. I saw `test accuracy: 1.0` and `saved model.joblib`.

3. **Write the API** ([[FastAPI]]). The model is loaded once at start-up. The request body is declared as a class, so a missing or non-positive number is refused before your code runs. The model's chances come from `predict_proba`, which for three classes uses a [[Softmax]].

   **`main.py`**
   ```python
   import joblib
   from fastapi import FastAPI
   from pydantic import BaseModel, Field

   SPECIES = ["setosa", "versicolor", "virginica"]
   model = joblib.load("model.joblib")                       # loaded once, when the server starts
   app = FastAPI(title="Iris server")

   class Flower(BaseModel):                                   # centimetres; every value must be above 0
       sepal_length: float = Field(gt=0)
       sepal_width: float = Field(gt=0)
       petal_length: float = Field(gt=0)
       petal_width: float = Field(gt=0)

   @app.get("/health")
   def health():
       return {"status": "ok"}

   @app.post("/predict")
   def predict(f: Flower):
       chances = model.predict_proba([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]])[0]
       best = int(chances.argmax())
       return {"species": SPECIES[best], "chances": {s: round(float(c), 3) for s, c in zip(SPECIES, chances)}}
   ```

4. **Test the API without starting a server** ([[pytest]]). The test client calls the app inside Python.

   **`test_api.py`**
   ```python
   from fastapi.testclient import TestClient
   from main import app

   c = TestClient(app)

   def test_health():
       assert c.get("/health").json() == {"status": "ok"}

   def test_setosa():
       r = c.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
       assert r.status_code == 200 and r.json()["species"] == "setosa"

   def test_virginica():
       r = c.post("/predict", json={"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.4, "petal_width": 2.1})
       assert r.json()["species"] == "virginica"

   def test_chances_add_up_to_one():
       r = c.post("/predict", json={"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 4.1, "petal_width": 1.0})
       assert abs(sum(r.json()["chances"].values()) - 1) < 0.01

   def test_bad_input_is_refused():
       assert c.post("/predict", json={"sepal_length": -1, "sepal_width": 3, "petal_length": 1, "petal_width": 1}).status_code == 422
       assert c.post("/predict", json={"sepal_length": 5}).status_code == 422
   ```
   Run `pytest -q`. I saw `5 passed` (in about 2 seconds).

5. **Run the real server and call it.** In one terminal:
   ```
   uvicorn main:app --port 8000
   ```
   In a second terminal, with this small script:

   **`call.py`**
   ```python
   import httpx

   r = httpx.post("http://localhost:8000/predict",
                  json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
   print(r.status_code, r.json())
   ```
   Run `python call.py`. I saw (the same call, made inside my own test script):
   ```
   200 {'species': 'setosa', 'chances': {'setosa': 0.976, 'versicolor': 0.024, 'virginica': 0.0}}
   ```
   A negative value gave status 422 with the message `Input should be greater than 0`, and `http://localhost:8000/docs` answered 200 with the automatic documentation page.

6. **Write the web page** ([[Streamlit]]). Each slider is a Python variable. Every time one moves, Streamlit reruns the script, which calls the API again. The address of the API comes from an environment variable so the same file works on your machine and in a container.

   **`app.py`**
   ```python
   import os
   import httpx
   import streamlit as st

   API = os.environ.get("API_URL", "http://localhost:8000")   # where the FastAPI server lives

   st.title("Which iris is it?")
   sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
   sw = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
   pl = st.slider("Petal length (cm)", 1.0, 7.0, 4.3)
   pw = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

   try:
       r = httpx.post(f"{API}/predict", json={"sepal_length": sl, "sepal_width": sw, "petal_length": pl, "petal_width": pw}, timeout=5)
       r.raise_for_status()
       out = r.json()
       st.success(f"Prediction: {out['species']}")
       st.bar_chart(out["chances"])
   except httpx.HTTPError as e:
       st.error(f"Could not get a prediction from the API: {e}")
   ```
   With the server from step 5 still running, start the page with `streamlit run app.py` and open the address it prints. **I did not open it in a browser.** I ran Streamlit's test tool against the live server instead: the default sliders gave `Prediction: versicolor`; moving them to 5.1, 3.5, 1.4, 0.2 gave `setosa`; moving them to 6.9, 3.1, 5.4, 2.1 gave `virginica`. With the server stopped, the page showed `Could not get a prediction from the API: [WinError 10061] No ...` (the message is the system's wording for "connection refused").

7. **Package it** ([[Docker]]). Two small images, one per service, and a file that starts both. The API image trains the model while it is built, so the model file is inside the image.

   **`requirements-api.txt`**
   ```
   fastapi==0.128.8
   uvicorn==0.39.0
   scikit-learn==1.6.1
   joblib==1.5.3
   ```
   **`requirements-ui.txt`**
   ```
   streamlit==1.50.0
   httpx==0.28.1
   ```
   **`Dockerfile.api`**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements-api.txt .
   RUN pip install --no-cache-dir -r requirements-api.txt
   COPY train.py main.py ./
   RUN python train.py
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
   **`Dockerfile.ui`**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements-ui.txt .
   RUN pip install --no-cache-dir -r requirements-ui.txt
   COPY app.py .
   ENV API_URL=http://api:8000
   CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
   ```
   **`compose.yaml`**
   ```yaml
   services:
     api:
       build:
         context: .
         dockerfile: Dockerfile.api
       ports:
         - "8000:8000"
     ui:
       build:
         context: .
         dockerfile: Dockerfile.ui
       ports:
         - "8501:8501"
       depends_on:
         - api
   ```
   The requirements are listed before the code is copied, so editing `main.py` or `app.py` does not make Docker reinstall the libraries (the cache rule in [[Docker]]). To run it, `docker compose up --build`, then open `http://localhost:8501`. **I ran this step** from files extracted out of this note: `docker compose build`, then `docker compose up -d`, then the calls shown in the result section, then `docker compose down --rmi local` to clean up. Both images built and both containers stayed up.

## What went wrong, and the fix
- **FastAPI would not import in my main Python.** The error came from a clash between an old FastAPI and a newer Pydantic that were both installed. The fix was a fresh virtual environment (step 1), which is also the reason that step exists.
- **Streamlit's test tool does not exist in older versions.** My main Python had Streamlit 1.25.0, which lacks it. The fresh environment has 1.50.0, which has it.
- **The page needs the API running.** Without it the page used to be the first thing to break, so `app.py` catches the connection error and shows a message instead of a traceback.

## What I did not run
- **The page in a real browser**, also inside Docker: I checked that port 8501 answers 200 and Streamlit's health check says ok, and that the page's container can reach the API, but I did not look at the drawn page.
- **A rebuild after editing code**, so the layer-cache claim in step 7 is the Docker rule, not something I timed here.
- **The Streamlit page in a real browser.** I used Streamlit's test tool, which runs the script and reads its output but does not draw the page.
- **Anything over a network other than my own machine**, and any load or speed test. The 25 ms figure is one call.

## Concepts used
- [[Logistic Regression]] · [[Classification]] · [[Softmax]] · [[Supervised Learning]] · [[Evaluation Metrics]]
- [[FastAPI]] · [[Streamlit]] · [[Docker]] · [[pytest]] · [[Virtual Environments]] · [[Command Line]] · [[Python]]
- [[MLOps]] · [[Model Monitoring Lab]]
## Next ideas
- Edit `main.py`, rebuild, and time it to see the layer cache work.
- Add authentication and a rate limit before putting the API on the internet.
- Swap the model for something larger ([[PyTorch]]) and see how start-up time and the Docker image size change.
- Log every request, and watch how the predictions drift if the inputs change ([[Evaluation Metrics]]).

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-classic-ml What were the four "done" checks of the Iris Classifier Service?::(1) the model trains and is saved, (2) the API answers a request with a species and chances and refuses bad input, (3) the web page shows a prediction as the sliders move, (4) the two services are described so that `docker compose up` could start them.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What model and score did the project get?::A logistic regression on the 150-flower iris data set. It scored 1.0 accuracy on 45 held-out flowers, a toy data set, so this says little.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What did the API answer for a setosa request, and how fast?::`setosa` with chances 0.976 / 0.024 / 0.0, in about 25 ms for one local call (one call, not a benchmark). A negative measurement was refused with status 422.

#flashcards/projects/concept #flashcards/topic/project-classic-ml How was the Streamlit page tested?::With Streamlit's own test tool against the real running server. It showed `versicolor` for the default sliders, then `setosa` and `virginica` after moving the sliders to typical values. With the API stopped it showed a clear error instead of crashing.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What did Docker show in this project?::Both images built and `docker compose up` started both containers (API image 577 MB, page image 751 MB). `/health` answered ok, the setosa request returned the same chances (one call, 33 ms), a negative value gave 422, and the page at `http://localhost:8501` answered 200. From inside the page's container, `http://api:8000/predict` returned `virginica` for 6.9, 3.1, 5.4, 2.1, so the container-to-container address works.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why did FastAPI not import in the main Python?::An old FastAPI and a newer Pydantic were both installed and clashed. The fix was a fresh virtual environment, which is the reason that step exists.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why did the Streamlit test tool not work at first?::The main Python had Streamlit 1.25.0, which lacks it. The fresh environment has 1.50.0, which has it.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why does `app.py` catch the connection error?::The page needs the API running, and without it the page used to be the first thing to break. Now it shows a message instead of a traceback.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What did the Iris project not run?::The page in a real browser (also inside Docker, only port 8501 returning 200 and the container-to-container call were checked), a rebuild after editing code (the layer-cache claim is the Docker rule, not timed), and anything over a network other than the local machine or any load test. The 25 ms figure is one call.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Which versions did the Iris project use?::Python 3.9, scikit-learn 1.6.1, joblib 1.5.3, FastAPI 0.128.8, uvicorn 0.39.0, Streamlit 1.50.0, httpx 0.28.1, pytest 8.4.2 and Docker 28.0.4 with base image `python:3.9-slim`.
