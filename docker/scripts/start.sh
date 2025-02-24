#!/bin/bash

set -e  # Exit the script if any statement returns a non-true return value

# ---------------------------------------------------------------------------- #
#                          Function Definitions                                #
# ---------------------------------------------------------------------------- #

# Execute script if exists
execute_script() {
    local script_path=$1
    local script_msg=$2
    if [[ -f ${script_path} ]]; then
        echo "${script_msg}"
        bash ${script_path}
    fi
}

# Setup ssh
setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh

        if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
            ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -q -N ''
            echo "RSA key fingerprint:"
            ssh-keygen -lf /etc/ssh/ssh_host_rsa_key.pub
        fi

        if [ ! -f /etc/ssh/ssh_host_dsa_key ]; then
            ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key -q -N ''
            echo "DSA key fingerprint:"
            ssh-keygen -lf /etc/ssh/ssh_host_dsa_key.pub
        fi

        if [ ! -f /etc/ssh/ssh_host_ecdsa_key ]; then
            ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -q -N ''
            echo "ECDSA key fingerprint:"
            ssh-keygen -lf /etc/ssh/ssh_host_ecdsa_key.pub
        fi

        if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
            ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ''
            echo "ED25519 key fingerprint:"
            ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
        fi

        service ssh start

        echo "SSH host keys:"
        for key in /etc/ssh/*.pub; do
            echo "Key: $key"
            ssh-keygen -lf $key
        done
    fi
}

# Download specified repo and install dependencies 
download_repo() {
    local repo_url=$1
    local repo_dir=$2
    local python_version=$3

    if [[ -n "${GIT_TOKEN}" ]]; then
        # Assumes REPO_URL is in HTTPS format
        repo_url=$(echo "${repo_url}" | sed -e "s#https://#https://${GIT_TOKEN}@#")
    fi

    if [[ ! -d ${repo_dir} ]]; then
        echo "Cloning repo..."
        git clone "${repo_url}" "${repo_dir}"
    fi

    cd "${repo_dir}"

    if [[ -f "uv.lock" ]]; then
        echo "Installing dependencies with uv..."
        uv python install "${python_version}"
        uv venv
        source .venv/bin/activate
        uv sync
    fi
}

# Set git username and email
setup_git() {
    local git_email=$1
    local git_name=$2

    echo "Setting up git..."
    git config --global user.email "${git_email}"
    git config --global user.name "${git_name}"
}

# Export env vars
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

download_hf() {
    local repo_dir="$1"
    
    cd "${repo_dir}"
    source .venv/bin/activate
    HF_AUTH_TOKEN="${HF_AUTH_TOKEN}" python -m src.stages.download_hf --config ./params.yaml
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #

echo "Pod Started"

setup_ssh
export_env_vars
download_repo "$REPO_URL" "$REPO_DIR" "$PYTHON_VERSION"
setup_git "$GIT_EMAIL" "$GIT_NAME"
download_hf "$REPO_DIR"

execute_script "/post_start.sh" "Running post-start script..."

echo "Start script(s) finished, pod is ready to use."

sleep infinity