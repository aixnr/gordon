import os
import requests
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import pathlib
from dotenv import load_dotenv

load_dotenv(pathlib.Path.cwd() / "config.env", override=False)

model_chat = os.getenv("GORDON_MODEL_CHAT", "gpt-oss-20b")
model_embedding = os.getenv("GORDON_MODEL_EMBEDDING", "text-embedding-mxbai-embed-large-v1")
api_endpoint = os.getenv("GORDON_MODEL_ENDPOINT", "http://127.0.0.1:1234/v1")
api_key = os.getenv("GORDON_API_KEY", "dummy-key")

# announce to user
print(f"[*] API endpoint is {api_endpoint}")
print(f"[*] Using {model_embedding} for the embedding...")
print(f"[*] Using {model_chat} for the conversation...")


class LocalOpenAIEmbeddings(Embeddings):
    def __init__(self, model, base_url=api_endpoint, api_key=api_key):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def embed_documents(self, texts):
        results = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={"model": self.model, "input": text}  # single string
            )
            resp.raise_for_status()
            results.append(resp.json()["data"][0]["embedding"])
        return results

    def embed_query(self, text):
        return self.embed_documents([text])[0]


embeddings = LocalOpenAIEmbeddings(model=model_embedding)


# Instead of using models available through providers
# we use API provided through locally-run LM Studio.
# Note that the API is not protected, but ChatOpenAI() class
# requires 'api_key' (pydantic.SecretStr) to be provided
llm = ChatOpenAI(
        model=model_chat,
        temperature=0,
        api_key=SecretStr(api_key),
        base_url=api_endpoint
    )
