import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

@dataclass(frozen=True)
class Provider: 
    name: str
    env_var: str
    is_free: bool
    base_url: str | None
    model: str


PROVIDERS = [
    Provider("Groq","GROQ_API_KEY",True,"https://api.groq.com/openai/v1","openai/gpt-oss-20b"),
    Provider("OpenAI","OPENAI_API_KEY",False, None,"gpt-4o-mini"),
]

def select_provider() -> Provider:
    for provider in PROVIDERS:
        api_key = os.getenv(provider.env_var)
        if api_key and api_key.strip():
            return provider

    raise RuntimeError(f"No provider found")


def client_and_model() -> tuple[OpenAI, str, Provider]:
    provider = select_provider()
    api_key = os.getenv(provider.env_var)
    if provider.base_url is None:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(api_key=api_key, base_url=provider.base_url)

    return client, provider.model, provider

