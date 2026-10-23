FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image ~2GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY faiss_index/ faiss_index/

ENV USER_AGENT=production-rag
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
