import os
import logging
from typing import Optional
import ollama

logger = logging.getLogger(__name__)

class FreeLLM:
    def __init__(self, model_name: str = "llama2:7b"):
        self.model_name = model_name
        try:
            ollama.list()
            logger.info(f"Ollama connected. Using model: {model_name}")
        except Exception as e:
            logger.error(f"Ollama not running. Please start Ollama first: {e}")
            raise
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = ollama.chat(model=self.model_name, messages=messages, options={"temperature": temperature})
            return response['message']['content']
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_with_context(self, query: str, context: str) -> str:
        system_prompt = "You are a helpful customer support assistant. Answer based ONLY on the provided context. If you don't know, say so. Keep answers concise."
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
        return self.generate(prompt, system_prompt)

_llm = None

def get_llm(model_name: str = "llama2:7b") -> FreeLLM:
    global _llm
    if _llm is None:
        _llm = FreeLLM(model_name)
    return _llm