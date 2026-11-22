---
tags: [programming, python, web, api, backend, deployment]
status: growing
level: 4
tier: 2
---

# FastAPI

**FastAPI** is a Python tool for building a web **API**: a program that other programs (a web page, a phone app, another service) can talk to over the internet by sending requests and getting answers back.

## 🟢 1. Simple (age 12)
Think of a restaurant. You do not walk into the kitchen. You give your order to a waiter, and the waiter brings back the food. An API is the waiter for a computer program: other programs hand it a request, and it hands back an answer. FastAPI helps you build that waiter, and it checks that every order is written properly before it reaches the kitchen.

**Example:** you trained a small model that guesses a flower's species. With FastAPI you wrap it so that anyone can send the flower's measurements and get "setosa" back in a moment, without having to install your model.

## 🔵 2. How it works
1. You write ordinary [[Python]] functions. Each function answers one kind of request, for example "give me item number 7" or "predict this flower".
2. You mark each function with a **route**: a web address and a method (`GET` to read something, `POST` to send something).
3. You describe the data you expect with **type hints** (`item_id: int`) or with a small class of fields. FastAPI uses these to check every request.
4. If a request is wrong (a word where a number should be, a missing field), FastAPI answers with an error (code 422) that says what is wrong. Your function is never called with bad data.
5. If it is fine, your function runs and its return value is sent back as JSON, the common text format for data.
6. FastAPI also builds a **documentation page** automatically (at `/docs`), where you can try every route in the browser.
7. A small server program (uvicorn) runs the app and listens for requests. To ship it, you usually put it in a [[Docker]] container.
8. **The catch:** FastAPI gives you the doorway, not the answers. A slow model is still slow, and you must still think about who is allowed to call it and how many calls to accept.

## 🟠 3. Technical detail
- **Built from:** Starlette for the web part and Pydantic for checking data (from memory). Type hints do the work of validation, conversion and documentation.
- **Validation:** a request body is declared as a Pydantic model. A field can have rules (`Field(gt=0)` means "greater than 0"), and a broken rule gives a 422 answer that names the field (shown in the code below).
- **Automatic docs:** FastAPI builds an OpenAPI description of the routes (a machine-readable list), and `/docs` shows it as a page. The code below lists the routes in that description.
- **Testing without a server:** `TestClient` calls the app directly inside Python, so you can check routes with ordinary tests ([[pytest]]).
- **Serving a model:** load the model once when the app starts, and call it inside the route ([[PyTorch]], [[TensorFlow]], [[Fine-tuning]]). Heavy models are slow and may need many workers or a [[GPU]].
- **Pairs with:** [[Streamlit]] (a quick front end that calls the API) and [[Docker]] (a way to package and run it anywhere).
- **Related ideas:** [[Tool Calling]] and [[Model Context Protocol]] are ways for language models to call APIs like this one.

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

app = FastAPI(title="tiny model server")

class Flower(BaseModel):                                   # the request body: declared once, checked automatically
    sepal_length: float = Field(gt=0)
    petal_length: float = Field(gt=0)

@app.get("/items/{item_id}")
def read_item(item_id: int, shout: bool = False):          # type hints decide how the path and query are parsed
    return {"item_id": item_id, "text": "HELLO" if shout else "hello"}

@app.post("/predict")
def predict(f: Flower):
    return {"species": "setosa" if f.petal_length < 2.5 else "other"}

c = TestClient(app)                                        # calls the app in-process: no server, no network
for method, url, body in (("GET", "/items/7?shout=true", None), ("GET", "/items/seven", None),
                          ("POST", "/predict", {"sepal_length": 5.1, "petal_length": 1.4}),
                          ("POST", "/predict", {"sepal_length": 5.1}),
                          ("POST", "/predict", {"sepal_length": 5.1, "petal_length": -3})):
    r = c.request(method, url, json=body)
    detail = r.json()["detail"][0]["msg"] if r.status_code == 422 else r.json()
    print(f"{method:4s} {url:22s} {str(body):45s} -> {r.status_code} {detail}")
print("automatic docs page /docs:", c.get("/docs").status_code, "| OpenAPI description lists", sorted(c.get("/openapi.json").json()["paths"]))
hand = '''def predict(data):
    if not isinstance(data, dict): return 422, "body must be an object"
    for k in ("sepal_length", "petal_length"):
        if k not in data: return 422, k + " is required"
        if not isinstance(data[k], (int, float)): return 422, k + " must be a number"
        if data[k] <= 0: return 422, k + " must be positive"
    return 200, "setosa" if data["petal_length"] < 2.5 else "other"'''
model = '''class Flower(BaseModel):
    sepal_length: float = Field(gt=0)
    petal_length: float = Field(gt=0)'''
print("checking two fields by hand:", len(hand.splitlines()), "lines of code; declaring them:", len(model.splitlines()), "lines")
```

## 🔴 4. Deep / formal
**What an API contract is.** A web API is an agreement: for a given address and method, the caller sends data in an agreed shape, and the server returns data in an agreed shape with a status code. FastAPI writes the contract once, in type hints, and uses it three ways: to **check** requests, to **convert** text into numbers and dates, and to **document** the routes.

**Status codes used above:** 200 means success, and 422 means the request was understood but its data is not valid (from memory of the standard meanings).

**Trade-offs:**
- **Validation is a safety net, not security.** Anyone on the internet can call a public route, so authentication and limits on request size and rate are separate jobs.
- **Declared types make errors early and clear,** at the cost of describing every input.
- **One process serves many requests by taking turns,** so a long model call can block others unless it runs in separate workers (from memory).

**What came before, and how this improved on it:**
- **Before:** web code where each handler read the raw request and checked every field by hand, as in older Python frameworks such as Flask (2010, Armin Ronacher; from memory, not checked), with documentation written separately.
- **Its limit:** the hand-written checks are long, easy to forget, and drift away from the documentation.
- **What changed:** declare the shape of the data once with type hints and Pydantic models, and let the framework check requests and generate the documentation. FastAPI was created by Sebastian Ramirez and first released around 2018 (from memory, not checked).
- **How much better:** in the code below, five requests got the right status each time (200, 422 for a word instead of a number, 422 for a missing field, 422 for a negative value, 200), with a message that names the problem. Checking two fields by hand takes 7 lines, and declaring them takes 3. The `/docs` page answers 200 and the generated description lists both routes. I did not measure speed; the project's claims about speed are from memory and untested here.

**Check it yourself:** run the code in level 3 above (it needs `pip install fastapi httpx`). It prints every number quoted in this block.

## Related
- [[Python]] · [[Streamlit]] · [[Docker]] · [[pytest]] · [[Virtual Environments]] · [[PyTorch]] · [[TensorFlow]] · [[Tool Calling]] · [[Model Context Protocol]] · [[GPU]] · [[Iris Classifier Service]]
- [[MLOps]] · [[ONNX]] · [[Model Monitoring Lab]]
## Sources
- FastAPI documentation, https://fastapi.tiangolo.com (routes, Pydantic models and the automatic `/docs` page: from memory, not reopened this session).
- The code and the numbers were run for this note with FastAPI 0.128.8, Pydantic 2.13.5 and Starlette 0.49.3 in a fresh virtual environment.
- No image: I found no free, on-topic picture on Wikimedia Commons, so I added none.
- Names and dates marked "from memory" are not checked.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/web-apps API:::A waiter for a computer program. Other programs hand it a request and it hands back an answer.

#flashcards/programming/reverse #flashcards/topic/web-apps FastAPI:::A Python framework for building that waiter. It checks that every request is written properly before your code runs.

#flashcards/programming/concept #flashcards/topic/web-apps What are a route, GET and POST?::A route is a web address plus a method. `GET` reads something and `POST` sends something.

#flashcards/programming/concept #flashcards/topic/web-apps How does FastAPI check a request?::From type hints (`item_id: int`) or a small class of fields. A wrong request (a word where a number should be, a missing field) gets an error with code 422 saying what is wrong, and your function is never called with bad data.

#flashcards/programming/concept #flashcards/topic/web-apps What does your FastAPI function return, and in what form does it travel?::Its return value is sent back as JSON, the common text format for data.

#flashcards/programming/concept #flashcards/topic/web-apps Where does FastAPI show its automatic documentation?::At `/docs`, where you can try every route in the browser. It is built from an OpenAPI description (a machine-readable list of routes).

#flashcards/programming/concept #flashcards/topic/web-apps What do uvicorn and Docker do for a FastAPI app?::Uvicorn is the small server program that runs the app and listens for requests. Docker packages it so it runs anywhere.

#flashcards/programming/example #flashcards/topic/web-apps How do you declare a rule such as "greater than 0" on a field?::With a Pydantic model and `Field(gt=0)`. A broken rule gives a 422 answer that names the field.

#flashcards/programming/concept #flashcards/topic/web-apps How can you test routes without starting a server?::Use `TestClient`, which calls the app directly inside Python, so ordinary tests (pytest) work.

#flashcards/programming/concept #flashcards/topic/web-apps How should a model be served from FastAPI?::Load the model once when the app starts and call it inside the route. Heavy models are slow and may need many workers or a GPU.

#flashcards/programming/cloze #flashcards/topic/web-apps FastAPI gives you the doorway, ==not the answers==: a slow model is still slow, and you must still think about who may call it and how many calls to accept.

#flashcards/programming/reverse #flashcards/topic/web-apps Built from:::Starlette for the web part and Pydantic for checking data (from memory in the note). Type hints do the work of validation, conversion and documentation.
