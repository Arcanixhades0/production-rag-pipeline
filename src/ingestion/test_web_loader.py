import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ingestion.web_loader import load_websites


documents = load_websites()

print()

print(f"Loaded {len(documents)} web documents")

for doc in documents:

    print("=" * 60)

    print(doc.metadata)

    print()

    print(doc.page_content[:500])