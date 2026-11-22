---
tags: [vector-database, embedding, RAG, LLM, search, HNSW]
status: growing
level: 4
tier: 3
---

# Vector Database

## 🟢 1. Simple (age 12)
A vector database is a **library that files things by meaning instead of by name**. Ask it "something like this" and it hands back the closest matches.

**Example:** In a normal library you look up a book by its exact title. In this library you say "I want a book about friendly dragons", and it brings you the books that *feel* closest to that, even if "friendly dragons" is not in any title.

## 🔵 2. How it works
1. Every item (a paragraph, a picture, a song) is turned into a [[Vector]] by an [[Embedding]] model. Similar items get similar numbers.
2. The database **stores** each vector, plus the original text and labels like "source" or "date".
3. When you ask a question, your question is turned into a vector with the **same** model.
4. The database finds the stored vectors **closest** to yours (see [[Cosine Similarity]]) and returns the original items.
5. Checking every item one by one is too slow for millions of items, so the database builds a **shortcut map** (an index) that finds close neighbours without looking at everything. The answer may very rarely miss the true best, but it is much faster.

This is the "library" that [[RAG]] uses: find the right pages first, then let the [[LLM]] read them.

![[vector-database-diagram.png]]
*Pipeline: text, images or sound go through an embedding model; the embeddings are indexed, and a search engine looks them up. Image: Jonas R L Goncalves, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vector_database_diagram.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
Core job: **k-nearest-neighbour (k-NN) search** over $N$ vectors of dimension $d$.

- **Exact (brute force / "flat"):** compare the query to all $N$ vectors, cost $O(Nd)$. Perfect results; fine up to about hundreds of thousands of vectors, or with a GPU.
- **Approximate (ANN):** trade a little recall for big speed. Main index families:

| Index | Idea | Notes |
|---|---|---|
| **HNSW** | layered graph of neighbours, greedy walk from coarse to fine layers | fast and accurate, high memory; common default |
| **IVF** | cluster the vectors, only search the few nearest clusters | less memory, needs training |
| **PQ** (product quantization) | compress vectors into short codes | huge memory saving, some accuracy loss |
| **DiskANN** | graph index living on SSD | billion-scale on one machine |

Key metrics: **recall@k** (how many true top-k neighbours are returned), latency, and memory.

Features beyond the index: **metadata filtering** ("only documents from 2024"), **hybrid search** (vector + keyword BM25), updates and deletes, replication and sharding.

Options: libraries (FAISS, hnswlib, Annoy), purpose-built databases (Qdrant, Milvus, Weaviate, Pinecone), and extensions to normal databases (pgvector for Postgres, SQLite and Elasticsearch/OpenSearch vector support). For a small personal project, a library or `pgvector` is usually enough.

```python
import numpy as np
rng = np.random.default_rng(0)
db = rng.normal(size=(10_000, 64)).astype("float32")
db /= np.linalg.norm(db, axis=1, keepdims=True)        # normalise once -> dot product = cosine

def search(q, k=3):                                    # brute-force exact k-NN
    q = q / np.linalg.norm(q)
    scores = db @ q
    top = np.argsort(-scores)[:k]
    return top, scores[top]

print(search(db[42] + 0.1 * rng.normal(size=64)))      # item 42 should come back first
```

Set up for [[RAG]]: split documents into chunks, embed each chunk, store `(vector, text, metadata)`, then at question time embed the question and take the top-k chunks.

## 🔴 4. Deep / formal
**Problem:** given query $q$ and set $X\subset\mathbb{R}^d$, return $\arg\min_{x\in X}\mathrm{dist}(q,x)$ (top-$k$ in general), where dist is Euclidean, or cosine / inner product (equivalent to Euclidean on unit vectors, since $\lVert q-x\rVert^2=2-2\,q\cdot x$).

**Curse of dimensionality:** tree indexes (k-d tree) work well for $d\lesssim 20$ but degrade toward a linear scan as $d$ grows, because distances concentrate and few branches can be pruned. ANN methods accept an approximation instead: return a point within $(1+\varepsilon)$ of the true nearest distance.

**HNSW** (Malkov & Yashunin, 2016): builds a multi-layer proximity graph. Each point is assigned a maximum layer

$$
\ell=\big\lfloor-\ln(U)\cdot m_L\big\rfloor,\qquad U\sim\mathrm{Uniform}(0,1)
$$

**Where:**
- $\ell$ is the highest layer the point will appear on (0 is the bottom layer holding everything).
- $U$ is a random number between 0 and 1, drawn fresh for each point.
- $-\ln(U)$ is a positive number that is usually small and only occasionally large. **Matters:** most points get a low $\ell$ and very few reach the top, which makes the upper layers sparse "highways".
- $m_L$ is a scale setting: bigger means more points climb to higher layers.
- $\lfloor\cdot\rfloor$ rounds down to a whole layer number.
- **Matters:** layer sizes shrink geometrically, like a skip list, so a search can cross the map quickly on the sparse top layers and then refine on the dense bottom.

**Example (tiny numbers):** with $M=16$ neighbours per node, a common choice is $m_L=1/\ln16\approx0.36$. Three new points get random draws:
- $U=0.5$: $-\ln(0.5)=0.69$, times $0.36=0.25$, rounded down $=\mathbf{layer\ 0}$ (the bottom, most common).
- $U=0.01$: $-\ln(0.01)=4.6$, times $0.36=1.66$, rounded down $=\mathbf{layer\ 1}$.
- $U=0.0001$: $-\ln(0.0001)=9.2$, times $0.36=3.3$, rounded down $=\mathbf{layer\ 3}$ (rare).
- On average only $1/16$ of the points reach layer 1 or higher, so each layer up is about 16 times smaller. That is why the top layers are sparse "highways". Search enters at the top layer, greedily moves to the neighbour closest to the query until no improvement, drops a layer, and repeats; at layer 0 it keeps a candidate list of size `efSearch`. Expected search cost is about $O(\log N)$. Main parameters:
- `M`: neighbours per node (memory and recall up)
- `efConstruction`: build-time effort (build time and graph quality up)
- `efSearch`: query-time effort (latency and recall up)

![[hnsw.png]]
*HNSW search: start at the sparse top layer's entry point, walk greedily toward the query vector, then descend to denser layers to refine. Image: Rose electric, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Hierarchical_Navigable_Small_World_(HNSW).png), CC BY 4.0.*

**Product quantization** (Jégou et al., 2011): split a $d$-dimensional vector into $m$ sub-vectors, run k-means (usually 256 centroids) on each subspace, and store only the $m$ centroid IDs. A $d=768$ float vector (3072 bytes) becomes e.g. 96 bytes; distances are approximated with precomputed lookup tables.

**Trade-offs and caveats:**
- Recall vs speed vs memory is a three-way trade; always measure recall against brute force on *your* data.
- Graph indexes handle deletes and inserts less gracefully than flat indexes; heavy filtering can break graph connectivity and hurt recall.
- The index is only as good as the embedding model. If the model is changed, **all vectors must be recomputed**.
- Similarity is not relevance: top-k by cosine can return near-duplicates or miss exact keywords, which is why hybrid search and **reranking** are common.

**What came before, and how this improved on it:**
- **Before:** compare the query with every stored vector (exact search).
- **Its limit:** the work grows with the number of vectors, which is too slow for millions.
- **What changed:** an approximate index. The code below uses the simplest one, buckets around centre points, where only the closest buckets are searched. Real systems use clever graph or bucket indexes such as HNSW and IVF (names from memory).
- **How much better:** in the code below (20,000 vectors in 50 dimensions, 200 queries, top 10 wanted), the bucket search looked at 2,073 vectors on average instead of 20,000, which is 9.6 times fewer, and still found 98.2% of the true top 10. That is the trade: a little recall for a lot of speed. The data is made of clusters, which helps; on data with no structure the recall would be lower.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
n, d, k = 20000, 50, 100
centers = rng.standard_normal((k, d)) * 3
X = centers[rng.integers(0, k, n)] + rng.standard_normal((n, d))
Q = centers[rng.integers(0, k, 200)] + rng.standard_normal((200, d))
cent = X[rng.choice(n, 100, replace=False)]                                  # 100 buckets: pick 100 points as bucket centres
assign = np.argmin((X ** 2).sum(1)[:, None] - 2 * X @ cent.T + (cent ** 2).sum(1)[None], 1)
hits, scanned = 0, 0
for q in Q:
    true = set(np.argsort(((X - q) ** 2).sum(1))[:10])                       # exact: look at everything
    near = np.argsort(((cent - q) ** 2).sum(1))[:5]                          # approximate: only the 5 closest buckets
    ids = np.where(np.isin(assign, near))[0]; scanned += len(ids)
    best = ids[np.argsort(((X[ids] - q) ** 2).sum(1))[:10]]; hits += len(true & set(best))
print(f"{n:,} vectors, 200 queries, top 10 wanted")
print(f"exact search looks at all {n:,}; bucket search looks at {scanned / 200:,.0f} on average ({n / (scanned / 200):.1f}x fewer)")
print(f"it still finds {hits / 2000:.1%} of the true top 10")
```

## Related
- [[Vector]] · [[Cosine Similarity]] · [[Embedding]] · [[RAG]] · [[LLM]] · [[GPU]] · [[k-NN]] · [[Clustering]] · [[Quantization]] · [[Big-O]] · [[Hash Tables]] · [[Sorting]] · [[Graphs]] · [[Heaps and Priority Queues]] · [[SQL]] · [[BM25]] · [[Retrieval Metrics]] · [[Vault Search with RAG]] · [[Chunking Strategies]] · [[Reranking]]
- [[Recommender Systems]]
- [[Two-Tower Model]]
## Sources
- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, 2016
- Jégou, Douze & Schmid, *Product Quantization for Nearest Neighbor Search*, 2011
- Johnson, Douze & Jégou, *Billion-scale similarity search with GPUs* (FAISS), 2017

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Vector database:::A library that files things by meaning instead of by name. Ask it "something like this" and it hands back the closest matches.

#flashcards/ml/formula #flashcards/topic/agents-rag Write how HNSW assigns a point its top layer and explain each symbol.
?
$$\ell=\big\lfloor-\ln(U)\cdot m_L\big\rfloor,\qquad U\sim\mathrm{Uniform}(0,1)$$
- $\ell$: the highest layer the point will appear on (0 is the bottom layer holding everything).
- $U$: a random number between 0 and 1, drawn fresh for each point.
- $-\ln(U)$: a positive number that is usually small and only occasionally large, so most points get a low $\ell$ and very few get a high one.
- $m_L$: a scale constant. With $M$ neighbours per node, a common choice is $m_L=1/\ln M$.

#flashcards/ml/cloze #flashcards/topic/agents-rag In HNSW, the top layers are ==sparse== highways and search goes ==coarse to fine==.

#flashcards/ml/example #flashcards/topic/agents-rag With $M=16$ ($m_L\approx0.36$), which layer do $U=0.5$, $0.01$ and $0.0001$ get?::U = 0.5: −ln 0.5 = 0.69, × 0.36 = 0.25, layer 0. U = 0.01: 4.6 × 0.36 = 1.66, layer 1. U = 0.0001: 9.2 × 0.36 = 3.3, layer 3 (rare). About 1/16 of the points reach layer 1 or higher.

#flashcards/ml/concept #flashcards/topic/agents-rag How does an HNSW search work?::Enter at the top layer, greedily move to the neighbour closest to the query until no improvement, drop a layer, and repeat. At the bottom layer, refine among the densest neighbours.

#flashcards/ml/image #flashcards/topic/agents-rag ![[vector-database-diagram.png]]
What does this picture show?
?
The pipeline: text, images or sound go through an embedding model, the embeddings are indexed, and a search engine looks them up.

#flashcards/ml/image #flashcards/topic/agents-rag ![[hnsw.png]]
What does this picture show?
?
HNSW search: start at the sparse top layer's entry point, walk greedily toward the query vector, then descend to denser layers to refine.
