#!/bin/bash
set -euo pipefail

gh_authenticate() {
    if [[ -n "${GIT_TOKEN}" ]]; then
        echo "🔐 Authenticating with GitHub..."
        
        echo "${GIT_TOKEN}" | gh auth login --with-token
        local auth_status=$?
        
        if [ $auth_status -ne 0 ]; then
            echo "❌ Authentication failed. Verifying:"
            echo "Token prefix: ${GIT_TOKEN:0:4}..."
            echo "Expiration: $(curl -sH "Authorization: token ${GIT_TOKEN}" https://api.github.com | jq -r '.expires_at')"
            exit 1
        fi
        
        gh auth setup-git
        git config --global credential.helper 'cache --timeout=7200'
    fi
}

setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "🔑 Configuring SSH access..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 600 ~/.ssh/authorized_keys

        for type in ed25519 rsa; do
            [[ -f "/etc/ssh/ssh_host_${type}_key" ]] || ssh-keygen -t $type -f "/etc/ssh/ssh_host_${type}_key" -q -N ""
        done
        
        service ssh start
    fi
}

download_repo() {
    local repo_url=$1
    local repo_dir=$2
    local python_version=$3

    local repo_identifier
    repo_identifier=$(echo "$repo_url" | sed -E 's#^https://github.com/([^/]+/[^/.]+).*#\1#')

    if [[ ! -d $repo_dir ]]; then
        echo "📥 Cloning ${repo_identifier}..."
        if ! gh repo clone "$repo_identifier" "$repo_dir" -- --depth=1 --recurse-submodules; then
            echo "⚠️ Cloning failed for '$repo_identifier'. Skipping this repo, but continuing the script..."
            return 0
        fi
    else
        echo "🔄 Validating existing repository..."
        cleanup_git_locks "$repo_dir"

        if [[ ! -d "$repo_dir/.git" ]]; then
            echo "⚠️ Corrupted repository detected - reinitializing..."
            if ! backup_and_reclone "$repo_dir" "$repo_identifier"; then
                echo "⚠️ Re-clone failed for '$repo_identifier'. Skipping this repo, but continuing..."
                return 0
            fi
        else
            echo "♻️ Updating repository..."
            cd "$repo_dir" || return 0

            if ! git fetch --depth=1; then
                echo "⚠️ 'git fetch' failed. Continuing with existing code..."
            fi

            if ! git reset --hard "@{u}"; then
                echo "⚠️ 'git reset' failed. Continuing with existing code..."
            fi

            if ! git lfs pull; then
                echo "⚠️ 'git lfs pull' failed. Continuing without LFS sync..."
            fi
        fi
    fi

    # Always set up environment after update attempts
    cd "$repo_dir" || return 0
    setup_virtualenv "$python_version"
    install_dependencies
}

cleanup_git_locks() {
    local repo_dir=$1
    
    if [[ -d "${repo_dir}/.git" ]]; then
        echo "Cleaning up any stale git lock files..."
        find "${repo_dir}/.git" -name "*.lock" -print -delete
    fi
}

backup_and_reclone() {
    local repo_dir=$1
    local repo_identifier=$2
    local backup_dir="${repo_dir}_backup_$(date +%s)"
    
    echo "🔄 Backing up to ${backup_dir}..."
    mv -f "$repo_dir" "$backup_dir"
    gh repo clone "$repo_identifier" "$repo_dir" -- --depth=1 --recurse-submodules
}

setup_virtualenv() {
    local python_version=$1
    echo "🐍 Creating Python ${python_version} environment..."
    
    rm -rf .venv
    uv venv --python="$python_version" .venv
    source .venv/bin/activate
}

install_dependencies() {
    echo "📦 Installing dependencies..."
    uv pip install --upgrade pip setuptools wheel

    # Verify critical files exist
    if [[ ! -f "pyproject.toml" && ! -f "requirements.txt" ]]; then
        echo "❌ Missing dependency files - checking backups..."
        restore_from_backup
    fi

    if [[ -f pyproject.toml ]]; then
        uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
    elif [[ -f requirements.txt ]]; then
        uv pip install -r requirements.txt
    fi
}

download_hf() {
    local repo_dir="$1"
    shift  # Remove repo_dir from arguments list
    
    echo "🤗 Downloading Hugging Face resources..."
    cd "${repo_dir}"
    source .venv/bin/activate
    
    if [[ $# -gt 0 ]]; then
        # Process each model individually
        for model in "$@"; do
            echo "📥 Downloading model: ${model}"
            local cmd=".venv/bin/python -m src.stages.download_hf --config ./params.yaml --model ${model}"
            HF_AUTH_TOKEN="${HF_AUTH_TOKEN}" ${cmd} || {
                echo "⚠️ Model download failed for ${model} - continuing with next model"
            }
        done
    else
        # No specific models provided, run default download
        local cmd=".venv/bin/python -m src.stages.download_hf --config ./params.yaml"
        HF_AUTH_TOKEN="${HF_AUTH_TOKEN}" ${cmd} || {
            echo "⚠️ Model download failed - continuing with existing files"
        }
    fi
}

setup_git() {
    [[ -n "$GIT_EMAIL" ]] && git config --global user.email "${GIT_EMAIL}"
    [[ -n "$GIT_NAME" ]] && git config --global user.name "${GIT_NAME}"
}

export_env_vars() {
    echo "🌐 Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

# ---------------------------------------------------------------------------- #
#                               Main Execution                                 #
# ---------------------------------------------------------------------------- #

echo "🚀 Starting Pod Initialization"
setup_ssh
gh_authenticate
setup_git
export_env_vars
download_repo "$REPO_URL" "$REPO_DIR" "$PYTHON_VERSION"
if [[ -n "${EXP_MODEL_ID:-}" ]]; then
    # Split on whitespace into an array of models
    read -r -a models <<< "$EXP_MODEL_ID"
    echo "ℹ️ Overriding default model download with: ${models[*]}"
    # Pass the models as separate arguments
    download_hf "$REPO_DIR" "${models[@]}"
else
    download_hf "$REPO_DIR"
fi

[[ -f "/post_start.sh" ]] && {
    echo "🔧 Running post-start script..."
    bash "/post_start.sh"
}

echo "✅ Initialization Complete - Pod Ready"
sleep infinity
