# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 7861


# Define environment variable
ENV LOG_PATH=/app/logs/
ENV CORPUS_PATH=/app/corpus/
ENV GRADIO_SERVER_NAME 0.0.0.0
ENV GRADIO_SERVER_PORT 7861

# put openai api key here
ENV OPENAI_API_KEY sk-xxxx

# Run your_script.py when the container launches
CMD ["python", "main.py"]
