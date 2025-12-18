# Use official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (gcc for python packages, curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better layer caching
COPY pyproject.toml ./

# Install dependencies first (this layer will be cached if dependencies don't change)
RUN pip install --no-cache-dir .

# Now copy the rest of the source code
COPY . .

# Reinstall in case there are local imports that need the full codebase
RUN pip install --no-cache-dir .

# Create necessary directories
RUN mkdir -p /app/data_uploads /app/models /app/figures

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
