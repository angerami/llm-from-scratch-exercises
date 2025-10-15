# LLM From Scratch - Exercises

Personal implementation of exercises from **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka. Building a GPT-like LLM in PyTorch from the ground up.

## About

Working through the book chapter by chapter, implementing:
- Text tokenization and data processing
- Attention mechanisms
- GPT architecture from scratch
- Model pretraining
- Fine-tuning for classification and instruction following

## Structure

```
├── chapter02/          # Text data processing
├── chapter03/          # Attention mechanisms
├── chapter04/          # GPT model implementation
├── chapter05/          # Pretraining
├── chapter06/          # Classification fine-tuning
├── chapter07/          # Instruction fine-tuning
├── requirements.txt    # Dependencies
└── test_setup.py       # Environment verification
```

## Setup

```bash
# Create conda environment
conda create -n llm-env python=3.11
conda activate llm-env

# Install dependencies
pip install -r requirements.txt

# Verify setup
python test_setup.py
```

## Book & Resources

- **Book:** [Manning - Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- **Official Repo:** [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- **Author:** Sebastian Raschka

## Progress

Track progress in commit history. Each commit follows format: `[ChX] Description`
