from setuptools import setup, find_packages

setup(
    name="free-rag-support-kb",
    version="0.1.0",
    description="FREE Intelligent Support Knowledge Base using RAG",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "ollama>=0.1.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
)