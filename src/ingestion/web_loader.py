from langchain_community.document_loaders import WebBaseLoader


def load_websites(file_path="data/web/urls.txt"):

    documents = []

    with open(file_path, "r", encoding="utf-8") as f:
        urls = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(f"\nFound {len(urls)} URLs")

    for url in urls:

        print(f"Loading: {url}")

        try:

            loader = WebBaseLoader(url)

            documents.extend(
                loader.load()
            )

        except Exception as e:

            print(f"Failed: {url}")

            print(e)

    return documents