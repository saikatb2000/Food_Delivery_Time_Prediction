FROM python:3.9

# Set working directory
WORKDIR /ml_project

# Copy all files
COPY . /ml_project

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# Run the application
CMD ["python", "app.py"]