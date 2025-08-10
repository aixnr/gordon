import argparse
import asyncio
import json
import os
import sys
from typing import List, Dict, Any

import aiohttp
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from .loadmodel import embeddings

def load_sources(json_path: str) -> List[Dict[str, Any]]:
    """
    Load sources JSON, normalize it so each entry has a single 'url' string.
    Supports 'url' field as string or list of strings.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        sources = json.load(f)
    if not isinstance(sources, list):
        raise ValueError("JSON root must be a list of objects")

    normalized_sources = []
    for source in sources:
        urls = source.get("url")
        if urls is None:
            print("[!] Warning: source missing 'url' field, skipping:", source)
            continue

        if isinstance(urls, str):
            urls = [urls]
        elif not isinstance(urls, list):
            print("[!] Warning: 'url' must be string or list, skipping:", source)
            continue

        for url in urls:
            entry = dict(source)
            entry["url"] = url
            normalized_sources.append(entry)

    return normalized_sources


async def fetch_page(session: aiohttp.ClientSession, url: str, timeout: int = 15) -> str:
    headers = {
        "User-Agent": "ingest-web/1.0 (+https://yourdomain.example)"
    }
    async with session.get(url, headers=headers, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.text()


def parse_extract(html: str, src: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Parse HTML with BeautifulSoup and extract blocks based on tags/selectors.
    Returns list of {"text": ..., "method": "tag|selector|fallback", "pattern": ...}.
    """
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, str]] = []

    tags = src.get("tags") or []
    selectors = src.get("selectors") or []

    # By tag name
    for tag in tags:
        for el in soup.find_all(tag):
            text = el.get_text(separator=" ", strip=True)
            if text:
                items.append({"text": text, "method": "tag", "pattern": tag})

    # By CSS selector
    for sel in selectors:
        for el in soup.select(sel):
            text = el.get_text(separator=" ", strip=True)
            if text:
                items.append({"text": text, "method": "selector", "pattern": sel})

    # Fallback to body if nothing found
    if not items:
        body = soup.body.get_text(separator=" ", strip=True) if soup.body else soup.get_text(separator=" ", strip=True)
        if body:
            items.append({"text": body, "method": "fallback", "pattern": "body"})

    return items


async def scrape_one(session: aiohttp.ClientSession, src: Dict[str, Any], semaphore: asyncio.Semaphore,
                     pause: float, loop: asyncio.AbstractEventLoop) -> List[Document]:
    """
    Fetch a single URL and extract documents (as langchain Documents).
    Runs BeautifulSoup parsing in a threadpool to avoid blocking the event loop.
    """
    url = src.get("url")
    if not url:
        print("[!] Skipping source with no url:", src)
        return []

    async with semaphore:
        try:
            html = await fetch_page(session, url)
        except Exception as e:
            print(f"[!] Failed to fetch {url}: {e}")
            return []

    # Parse and extract in threadpool
    try:
        extracted = await loop.run_in_executor(None, parse_extract, html, src)
    except Exception as e:
        print(f"[!] Failed to parse/extract {url}: {e}")
        return []

    docs: List[Document] = []
    for i, item in enumerate(extracted):
        metadata = {
            "source": url,
            "extract_method": item.get("method"),
            "extract_pattern": item.get("pattern"),
            "block_index": i,
        }
        docs.append(Document(page_content=item["text"], metadata=metadata))

    # optional polite pause (between releasing semaphore and returning)
    if pause and pause > 0:
        await asyncio.sleep(pause)

    return docs


async def scrape_sources_async(sources: List[Dict[str, Any]], concurrency: int = 5,
                               pause: float = 0.0, timeout: int = 15) -> List[Document]:
    """
    Scrape all sources concurrently (bounded by concurrency). Returns a list of Documents.
    """
    connector = aiohttp.TCPConnector(limit_per_host=concurrency)
    timeout_obj = aiohttp.ClientTimeout(total=None)  # per-request timeout handled in fetch_page
    headers = {"User-Agent": "ingest-web/1.0 (+https://github.com/aixnr)"}
    semaphore = asyncio.Semaphore(concurrency)
    docs: List[Document] = []
    loop = asyncio.get_event_loop()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj, headers=headers) as session:
        tasks = [
            asyncio.create_task(scrape_one(session, src, semaphore, pause, loop))
            for src in sources
        ]
        for task in asyncio.as_completed(tasks):
            try:
                res = await task
            except Exception as e:
                print("[!] Task error:", e)
                res = []
            docs.extend(res)

    return docs


def save_manifest(manifest: List[Dict[str, Any]], output_dir: str) -> None:
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Ingest web pages into FAISS vector store (concurrent).")
    parser.add_argument("json", nargs="?", default="web_sources.json", help="Path to JSON file describing sources")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for text splitting")
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds to wait after each request (politeness)")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent requests")
    parser.add_argument("--timeout", type=int, default=15, help="Per-request timeout (seconds)")
    parser.add_argument("--output", default="faiss_index", help="Directory to save FAISS index and manifest")
    args = parser.parse_args()

    try:
        sources = load_sources(args.json)
    except Exception as e:
        print("[!] Failed to load JSON:", e)
        sys.exit(1)

    print(f"[*] Scraping {len(sources)} web sources with concurrency={args.concurrency} ...")
    try:
        raw_docs = asyncio.run(scrape_sources_async(sources, concurrency=args.concurrency,
                                                    pause=args.pause, timeout=args.timeout))
    except KeyboardInterrupt:
        print("[!] Interrupted.")
        sys.exit(1)
    except Exception as e:
        print("[!] Error during scraping:", e)
        sys.exit(1)

    if not raw_docs:
        print("[!] No documents extracted. Exiting.")
        sys.exit(1)

    print(f"[*] Extracted {len(raw_docs)} blocks. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    docs = splitter.split_documents(raw_docs)

    index_path = args.output
    if os.path.exists(index_path):
        print(f"[*] Loading existing FAISS index from '{index_path}' and adding documents...")
        vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        vectordb.add_documents(docs)
    else:
        print("[*] Creating new FAISS index from documents...")
        vectordb = FAISS.from_documents(docs, embeddings)

    print(f"[*] Saving updated vector store at '{index_path}' ...")
    vectordb.save_local(index_path)

    # Build manifest mapping doc IDs to source and metadata
    manifest: List[Dict[str, Any]] = []
    for idx, d in enumerate(docs):
        md = d.metadata or {}
        entry = {
            "id": idx,
            "source": md.get("source"),
            "extract_method": md.get("extract_method"),
            "extract_pattern": md.get("extract_pattern"),
            "block_index": md.get("block_index"),
            "snippet": (d.page_content[:200] + "...") if len(d.page_content) > 200 else d.page_content,
        }
        manifest.append(entry)

    save_manifest(manifest, args.output)
    print(f"[*] Saved manifest with {len(manifest)} entries to '{os.path.join(args.output, 'manifest.json')}'")
    print("[*] Done.")


if __name__ == "__main__":
    main()
