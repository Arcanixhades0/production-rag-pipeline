import sqlite3

from langchain_core.documents import Document


def load_sql(db_path, table="models"):
    """
    Turn each table row into a short natural-language Document.

    Embedding models understand sentences far better than raw
    "BERT|Google|2018|340" rows, so we write each row out as text.
    """
    documents = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()

    for row in rows:
        text = (
            f"{row['name']} is a {row['architecture']} model released by "
            f"{row['organization']} in {row['year']} with about "
            f"{row['params_millions']:,} million parameters. {row['notes']}"
        )

        documents.append(Document(
            page_content=text,
            metadata={"source": f"sql://{table}/{row['name']}", "source_type": "sql"}
        ))

    print(f"Loaded {len(documents)} SQL rows")

    return documents
