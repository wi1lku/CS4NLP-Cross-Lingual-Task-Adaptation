from setuptools import setup, find_packages

setup(
    name="CS4NLP-Cross-Lingual-Task-Adaptation",
    version="0.1.0",
    install_requires=[
        "peft",
        "torch",
        "transformers",
        "wandb",
        "conllu",
        "pandas",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
)