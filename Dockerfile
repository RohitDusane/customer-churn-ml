# Step 1: Use an official Python image as a base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy requirements.txt to the container
COPY requirements.txt /app/requirements.txt

# Step 4: Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all application code to the container
COPY . /app

# Step 6: Expose the port your FastAPI app will run on
EXPOSE 8000

# Step 7: Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
