import requests
import os
import sys

# RunPod API endpoint
API_BASE_URL = "https://rest.runpod.io/v1"


def get_api_key():
    """Get the RunPod API key from environment variables"""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    return api_key


def get_pod_id():
    """Get the pod ID from environment variables"""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print("Error: RUNPOD_POD_ID environment variable not set")
        sys.exit(1)
    return pod_id


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
    """Main function to delete the pod"""
    print("Starting pod termination script...")

    # Get API key
    api_key = get_api_key()

    # Get pod ID
    pod_id = get_pod_id()
    print(f"Pod ID from environment variable: {pod_id}")

    # Delete the pod
    delete_pod(api_key, pod_id)

    print("Termination complete. This pod will be deleted shortly.")


if __name__ == "__main__":
    main()
