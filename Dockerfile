FROM python:3.11-slim
WORKDIR /app

# Copy dependency manifests first for better layer caching, then the source.
# requirements.txt installs the package itself via `-e .`, so pyproject.toml
# and the source tree must be present before pip runs.
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "frag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
