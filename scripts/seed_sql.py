"""Creates data/sql/ml_models.db, a small structured source for the RAG pipeline."""
import sqlite3

from src.config import SQL_DB

MODELS = [
    ("BERT", "Google", 2018, 340, "encoder", "Bidirectional transformer pretrained with masked language modeling; strong for classification and extraction."),
    ("GPT-2", "OpenAI", 2019, 1500, "decoder", "Autoregressive language model; showed zero-shot task transfer from scale."),
    ("T5", "Google", 2019, 11000, "encoder-decoder", "Frames every NLP task as text-to-text."),
    ("GPT-3", "OpenAI", 2020, 175000, "decoder", "Popularised few-shot in-context learning."),
    ("CLIP", "OpenAI", 2021, 400, "dual-encoder", "Learns a shared image-text embedding space with contrastive training."),
    ("Stable Diffusion", "Stability AI", 2022, 890, "latent diffusion", "Text-to-image diffusion model operating in a VAE latent space."),
    ("Whisper", "OpenAI", 2022, 1550, "encoder-decoder", "Speech recognition model trained on 680k hours of weakly supervised audio."),
    ("LLaMA 2", "Meta", 2023, 70000, "decoder", "Open-weight chat models released with a commercial-use licence."),
    ("Mistral 7B", "Mistral AI", 2023, 7300, "decoder", "Uses grouped-query and sliding-window attention for efficient inference."),
    ("all-MiniLM-L6-v2", "Sentence-Transformers", 2021, 22, "encoder", "Small sentence embedding model producing 384-dimensional vectors; used in this project."),
    ("Gemma 3 4B", "Google", 2025, 4000, "decoder", "Small open model that runs locally through Ollama."),
    ("Qwen3.5 9B", "Alibaba", 2026, 9000, "decoder", "Open-weight model used as this project's local generator."),
]


def main():
    SQL_DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(SQL_DB) as conn:
        conn.execute("DROP TABLE IF EXISTS models")
        conn.execute("""
            CREATE TABLE models (
                name TEXT PRIMARY KEY,
                organization TEXT,
                year INTEGER,
                params_millions INTEGER,
                architecture TEXT,
                notes TEXT
            )
        """)
        conn.executemany("INSERT INTO models VALUES (?, ?, ?, ?, ?, ?)", MODELS)

    print(f"Wrote {len(MODELS)} rows to {SQL_DB}")


if __name__ == "__main__":
    main()
