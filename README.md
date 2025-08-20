# gordon

Chat with documents and/or web pages with locally-hosted LLM chat model.
Document ingestion also uses local model.
The tested local model provider is LM Studio.

```bash
# setup the project
# final venv size on disk: 276.6 MiB
uv venv venv
source venv/bin/activate
uv pip install .

# ingest PDF documents
ingest-doc [documents/]

# ingest web sources
ingest-web [web_sources.json]

# chat with document
query
```

By default, running `ingest-{doc,web}` creates a directory `faiss_index` that stores the embedding vector, and by default, `query` reads data from that directory.
The `ingest-{doc,web}` can be run sequentially as it will append data to the `faiss_index`.
Note that the `ingest-web` script uses user-agent `"ingest-web/0.1.0 (+https://github.com/aixnr/gordon)"`.

Environmental variables (loaded from `config.env` by default; rename `config.example.env` to `config.env` after cloning this repo):

| Key                      | Default value
| :--                      | :--
| `GORDON_MODEL_CHAT`      | `gpt-oss-20b`
| `GORDON_MODEL_EMBEDDING` | `text-embedding-mxbai-embed-large-v1`
| `GORDON_MODEL_ENDPOINT`  | `http://127.0.0.1:1234/v1` (LM Studio locally)
| `GORDON_API_KEY`         | `dummy-key`
| `GORDON_CRAWL_DEPTH`     | 0, pass this when running `ingest-web` to crawl links

Tested with chat models `gpt-oss-20b` (from [Unsloth](https://huggingface.co/unsloth/gpt-oss-20b), quantized, `Q5_K_M` 10 GiB on disk, **11.3 GiB** VRAM) and `qwen3-14b` (quantized, `Q5_K_M` 9.8 GiB on disk, **10.4 GiB** VRAM).
GPU was AMD Radeon RX 7090 XT 20 GiB VRAM on Arch linux.

The content of the `web_sources.json` may look like this:

```json
[
  {
    "url": [
      "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    ],
    "selectors": [".mw-content-container"],
    "tags": ["p"]
  }
]
```

The `selectors` and `tags` can be defined together or only one of them, depending on the use case.
This information is passed to `BeautifulSoup` for scraping.
It falls back to scraping `body` if the pattern is not found.


## Rationale

Just an evening of vide-coding to come up with minimal yet modular _enough_ RAG implementation with fully offline deployment using local models.
Secondly, I needed some help reading a pile of journal articles and I figured a local RAG at the terminal would suffice.
An interactive reading session with a RAG does help to enhance comprehension and experience.
This project is named after [Gordon Edgley](https://skulduggery.fandom.com/wiki/Gordon_Edgley) from the novel series [Skulduggery Pleasant](https://en.wikipedia.org/wiki/Skulduggery_Pleasant), whom Stephanie Edgley often consulted with during her critical moments in the series.
Even though Gordon was merely a talking stone after his demise, he was essentially a Wikipedia that offered Stephanie guidance and advice.

License: MIT.

LLMs that assisted with the coding: GPT-4.1 Mini (OpenRouter), GPT-5, and gpt-oss-120b/20b.
