# Stage 1: Builder stage to install dependencies
FROM python:3.12-slim as builder

WORKDIR /usr/src/app

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file and install dependencies
COPY requirements.txt .
# The following RUN command is the key to the fix.
# It aggressively removes any existing 'google' packages to prevent namespace collisions,
# then installs the required 'google-generativeai' package first,
# and finally installs the rest of the application's dependencies.
RUN pip uninstall -y protobuf google-api-python-client google-cloud-storage google-auth google-cloud-core google-resumable-media google && \
    pip install --no-cache-dir --upgrade google-generativeai && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final stage with a clean environment
FROM python:3.12-slim

LABEL org.opencontainers.image.name="mem0/openmemory-mcp"

WORKDIR /usr/src/openmemory

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code
COPY . .

EXPOSE 8765
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765", "--reload"]
