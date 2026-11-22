---
tags: [programming, tooling, deployment, containers, devops]
status: growing
level: 4
tier: 3
---

# Docker

**Docker** is a tool that packs a program together with everything it needs (the right Python, the right libraries, its settings) into one sealed box called a **container**, so it runs the same way on any computer.

## 🟢 1. Simple (age 12)
Imagine you bake a cake that only works with one special oven. If you send the recipe to a friend with a different oven, the cake fails. Docker is like sending the cake **together with its own little oven**, already set up. Your friend just presses start, and the cake comes out the same as at your home.

**Example:** your machine-learning demo needs Python 3.9 and exact versions of five libraries. You put all of that in a Docker container. A teammate with a different computer, or a server in the cloud, runs the container and gets exactly your setup, with nothing to install by hand.

## 🔵 2. How it works
1. You write a short recipe file called a **Dockerfile**. It lists the steps: start from a base (for example one that already has Python), copy your files in, install the libraries, and say which command to run.
2. You **build** the recipe into an **image**: a frozen, read-only package of everything.
3. You **run** the image, and it becomes a live **container**: a sealed, isolated program with its own files, libraries and settings.
4. Many containers can run side by side on one computer, each in its own box, without disturbing each other.
5. Containers share the computer's core (its operating-system kernel), so they start in moments and are light (see the picture).
6. Each line of the Dockerfile makes a **layer**, and Docker remembers each layer. If a line and everything before it has not changed, Docker reuses the stored layer instead of redoing the work.
7. You can send an image to a shared store (a **registry**) so that other people or servers can download and run it.
8. **The catch:** a container is only as safe and as good as what you put in it. It does not make slow code fast, images can become large, and on Windows or Mac it runs inside a hidden Linux layer.

![[docker-vs-vm.jpg]]
*Virtual machines against containers. Left: each virtual machine carries its own full guest operating system on top of a hypervisor. Right: containers sit on one Docker engine and share the host's operating system, so each holds only the app and its libraries. Image: Drinkler, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Dockervsvm.jpg), CC BY-SA 4.0. I read the stacks and labels off the picture, not from the file text.*

## 🟠 3. Technical detail
- **Image and container:** an image is the template, a container is a running copy of it. Delete the container and the image is still there.
- **Dockerfile:** a text file of instructions, such as `FROM` (the base image), `COPY`, `RUN`, `CMD` (the default command) and `EXPOSE` (a port). A small example for a [[FastAPI]] app is below (a sketch; a real pair of Dockerfiles that I did build and run is in [[Iris Classifier Service]]).
- **Layers and the build cache:** a layer is rebuilt when its instruction or an earlier layer changed. So put the rarely-changing steps first (installing libraries) and the often-changing steps last (copying your code). The code below models this with assumed times.
- **Volumes and ports:** a container's own files disappear with it, so data you want to keep lives in a **volume**. A container's network port must be mapped to the host's to be reachable (from memory of the usual `-p` and `-v` options).
- **Compose:** a file that starts several containers together, for example an API container and a [[Streamlit]] container (from memory).
- **Machine learning:** images are used to share an exact training environment ([[PyTorch]], [[TensorFlow]], [[Virtual Environments]] do the same job at a smaller scale), and to deploy models behind an API. Using a [[GPU]] inside a container needs the host's GPU driver and a special runtime (from memory).
- **Compared with a virtual environment:** a [[Virtual Environments|virtual environment]] isolates Python libraries only. A container isolates the whole file system, the system libraries and the settings.
- **On this machine:** the Docker client was installed (version 28.0.4), but Docker Desktop's engine was not running, so I built and ran nothing here.

A sketch of a Dockerfile (not built or run; written from memory of the usual layout):

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```python
import shutil, subprocess
# A model of Docker's build cache: layers are built in order, and a changed layer forces every later layer to be rebuilt.
# The costs are ASSUMED round numbers (seconds), not measurements, and no image is built here.
cost = {"FROM base": 2, "COPY requirements.txt": 1, "RUN pip install": 60, "COPY . .": 1}
order_a = ["FROM base", "COPY . .", "RUN pip install"]                                   # copy everything first, then install the libraries
order_b = ["FROM base", "COPY requirements.txt", "RUN pip install", "COPY . ."]          # copy only the list of libraries, install, then copy the code
def total(order, edits):
    first = sum(cost[l] for l in order)                                                  # the first build runs every layer
    each = sum(cost[l] for l in order[order.index("COPY . ."):])                         # a code edit changes "COPY . .", so it and all later layers rerun
    return first + edits * each
for name, o in (("code copied first", order_a), ("libraries installed before the code is copied", order_b)):
    print(f"{name:48s}: first build {total(o, 0):3d} s; after 10 small code edits {total(o, 10):4d} s in total")
print("(assumed costs: base 2 s, copy list 1 s, install 60 s, copy code 1 s; nothing was built)")
print("docker command on this machine:", subprocess.run(["docker", "--version"], capture_output=True, text=True).stdout.strip() if shutil.which("docker") else "not installed")
```

## 🔴 4. Deep / formal
**What a container is.** A container is an ordinary process on the host, started with limits that hide the rest of the machine from it: its own view of the file system, its own network and its own list of processes, and a cap on the memory and processor it can use. The Linux kernel provides these limits (namespaces and control groups; from memory). This is why containers are light compared with virtual machines, which run a whole separate operating system.

**The cache rule.** Layers are built in order. Docker reuses a layer only if its instruction and all earlier layers are unchanged. So the rebuild cost after a change is the cost of the changed layer plus every layer after it.

**Trade-offs:**
- **Same environment everywhere** against **extra size and a new thing to learn.**
- **Sharing the host kernel** makes containers fast, but they are less strongly isolated than virtual machines (from memory).
- **Images pin old versions,** so you must rebuild to get security fixes.

**What came before, and how this improved on it:**
- **Before:** "it works on my machine": installing by hand on each computer, or shipping a whole **virtual machine**, which carries a full guest operating system for each app (see the picture).
- **Its limit:** hand installs drift apart between machines, and virtual machines are large and slow to start.
- **What changed:** package only the app and its libraries and share the host's operating system, in a standard image format with cached layers. Docker was presented to the public in 2013 (from memory, not checked).
- **How much better:** the code below is a model with **assumed** times and builds nothing. If the code is copied first and the libraries installed afterwards, every edit of the code reruns the install, so 10 small edits cost 673 s in total in the model. If the list of libraries is copied and installed first and the code last, the 10 edits cost 74 s. That is 9 times less, and it is a property of the layer order that you control, not of Docker alone. I measured no real build or container size, because the engine was not running.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block (the times are assumed, and the last line reports the Docker version if it is installed).

## Related
- [[Virtual Environments]] · [[FastAPI]] · [[Streamlit]] · [[Python]] · [[Command Line]] · [[Git]] · [[GPU]] · [[PyTorch]] · [[TensorFlow]] · [[pytest]] · [[Iris Classifier Service]]
- [[MLOps]] · [[ONNX]] · [[Model Monitoring Lab]]
## Sources
- Docker documentation, https://docs.docker.com (images, layers, the build cache, volumes, ports and Compose: from memory, not reopened this session).
- Image: [Dockervsvm.jpg](https://commons.wikimedia.org/wiki/File:Dockervsvm.jpg) by Drinkler, CC BY-SA 4.0.
- The layer-cache numbers come from a small model with assumed times, run for this note. No image was built and no container was run: the Docker client reported version 28.0.4 but its engine was not running.
- Names and dates marked "from memory" are not checked.

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/dev-tools Docker:::Sending a cake together with its own little oven. It packages your program with everything it needs so it runs the same on any computer.

#flashcards/programming/reverse #flashcards/topic/dev-tools Dockerfile:::A text recipe of steps: start from a base, copy your files in, install the libraries, and say which command to run.

#flashcards/programming/concept #flashcards/topic/dev-tools Image against container?::An image is a frozen, read-only template. A container is a live running copy of it. Delete the container and the image is still there.

#flashcards/programming/concept #flashcards/topic/dev-tools What do `FROM`, `COPY`, `RUN`, `CMD` and `EXPOSE` do in a Dockerfile?::`FROM` is the base image, `COPY` brings files in, `RUN` runs a build step, `CMD` is the default command, `EXPOSE` names a port.

#flashcards/programming/concept #flashcards/topic/dev-tools Why do containers start in moments and stay light?::They share the computer's core (the operating-system kernel) instead of carrying a full guest operating system like a virtual machine.

#flashcards/programming/concept #flashcards/topic/dev-tools How does the Docker build cache work, and how should you order a Dockerfile?::Each line makes a layer, and a layer is rebuilt only when its line or an earlier one changed. Put rarely changing steps first (installing libraries) and often changing steps last (copying your code).

#flashcards/programming/concept #flashcards/topic/dev-tools Where should data you want to keep live, and how is a port reached?::In a volume, because a container's own files disappear with it. A container's port must be mapped to the host's (from memory of the usual `-p` and `-v` options).

#flashcards/programming/reverse #flashcards/topic/dev-tools Registry:::A shared store where you send an image so other people or servers can download and run it.

#flashcards/programming/reverse #flashcards/topic/dev-tools Compose:::A file that starts several containers together, such as an API container and a Streamlit container (from memory).

#flashcards/programming/concept #flashcards/topic/dev-tools Docker against a virtual environment?::A virtual environment isolates Python libraries only. A container isolates the whole file system, the system libraries and the settings.

#flashcards/programming/concept #flashcards/topic/dev-tools What is the catch with Docker?::A container is only as good as what you put in it, it does not make slow code fast, images can become large, and on Windows or Mac it runs inside a hidden Linux layer.

#flashcards/programming/image #flashcards/topic/dev-tools ![[docker-vs-vm.jpg]]
What does this picture show?
?
Virtual machines against containers. Left: each virtual machine carries its own full guest operating system on top of a hypervisor. Right: containers sit on one Docker engine and share the host's operating system, so each holds only the app and its libraries. (From the note's caption: Drinkler, Wikimedia Commons. The note read the stacks off the picture.)

#flashcards/programming/cloze #flashcards/topic/dev-tools On the note's machine the Docker client (28.0.4) was installed but the engine was ==not running==, so nothing was built or run.
