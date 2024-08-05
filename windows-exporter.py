import os
import requests
import subprocess
import socket

# Configuration
windows_exporter_url = "https://github.com/prometheus-community/windows_exporter/releases/download/v0.25.1/windows_exporter-0.25.1-amd64.exe"
download_path = os.path.join(os.getenv("TEMP"), "windows_exporter.exe")
port = 9182


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('localhost', port)) == 0


def download_file(url, dest):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    else:
        print(f"Failed to download file: {response.status_code}")
        return False


def start_windows_exporter(path):
    print(f"Starting windows_exporter from {path}...")
    subprocess.Popen([path])


def get_process_id(process_name):
    result = subprocess.run(['tasklist', '/FI', f'IMAGENAME eq {process_name}'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if process_name in line:
            return int(line.split()[1])
    return None


if is_port_open(port):
    print(f"Port {port} is already in use. Exiting...")
else:
    if not os.path.exists(download_path):
        print("windows_exporter.exe not found, downloading...")
        if not download_file(windows_exporter_url, download_path):
            print("Failed to download windows_exporter. Exiting...")
            exit(1)

    start_windows_exporter(download_path)
    print("windows_exporter started successfully.")

    # Find and display the PID of the started process
    pid = get_process_id("windows_exporter.exe")
    if pid:
        print(f"windows_exporter.exe is running with PID: {pid}")
    else:
        print("Failed to find the PID of windows_exporter.exe.")
