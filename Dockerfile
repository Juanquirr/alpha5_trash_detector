# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the FastAPI app with Uvicorn
CMD ["/bin/bash"]
