import os
import subprocess

# Define the Docker image name
image_name = "anomaly_detector_app"

# Define the path to the application.yml file
config_path = os.path.abspath("application.yml")

# Build the Docker image
subprocess.run(["docker", "build", "-t", image_name, "."], check=True)

# Run the Docker container with volume mounting for the config file
subprocess.run([
    "docker", "run", "-d", "--name", "anomaly_detector_container",
    "-p", "5000:5000",
    "-v", f"{config_path}:/app/application.yml",
    image_name
], check=True)
