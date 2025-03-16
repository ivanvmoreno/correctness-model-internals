import requests
import socket
import os
import sys
import json

# RunPod API endpoint
API_BASE_URL = "https://rest.runpod.io/v1"


def get_api_key():
    """Get the RunPod API key from environment variables"""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    return api_key


def get_current_ip():
    """Get the current pod's public IP address using multiple methods"""
    try:
        response = requests.get("https://api.ipify.org", timeout=5)
        if response.status_code == 200:
            return response.text.strip()
    except Exception as e:
        print(f"Warning: Could not determine IP via external service: {e}")

    if os.environ.get("PUBLIC_IP"):
        return os.environ.get("PUBLIC_IP")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 1))  # Google's DNS server
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Warning: Could not determine IP via socket: {e}")

    print("Error: Could not determine current public IP")
    sys.exit(1)


def list_pods(api_key):
    """List all pods using the RunPod API"""
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(f"{API_BASE_URL}/pods", headers=headers)

    if response.status_code != 200:
        print(f"Error listing pods: {response.status_code} - {response.text}")
        sys.exit(1)

    return response.json()


def find_pod_id_by_ip(pods, ip):
    """Find the pod ID that matches the given IP address"""
    if not isinstance(pods, list):
        print(f"Error: Expected a list of pods but got {type(pods)}")
        print(f"Response data: {json.dumps(pods)[:500]}...")
        sys.exit(1)

    for pod in pods:
        if not isinstance(pod, dict):
            continue

        if pod.get("publicIp") == ip:
            pod_id = pod.get("id")
            if pod_id:
                return pod_id

    print(f"Error: No pod found with IP {ip}")
    print(
        f"Available pods: {json.dumps([p.get('publicIp') for p in pods if isinstance(p, dict)])}"
    )
    sys.exit(1)


def delete_pod(api_key, pod_id):
    """Delete a pod using the RunPod API"""
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Deleting pod with ID: {pod_id}")
    response = requests.delete(f"{API_BASE_URL}/pods/{pod_id}", headers=headers)

    if response.status_code != 200:
        print(f"Error deleting pod: {response.status_code} - {response.text}")
        sys.exit(1)

    print("Pod deletion initiated successfully")


def main():
    """Main function to find and delete the current pod"""
    print("Starting self-termination script...")

    # Get API key
    api_key = get_api_key()

    # Get current IP
    current_ip = get_current_ip()
    print(f"Current IP address: {current_ip}")

    # List all pods
    pods = list_pods(api_key)

    # Find pod ID by IP
    pod_id = find_pod_id_by_ip(pods, current_ip)
    print(f"Found pod ID: {pod_id}")

    # Delete the pod
    delete_pod(api_key, pod_id)

    print("Self-termination complete. This pod will be deleted shortly.")


if __name__ == "__main__":
    main()
