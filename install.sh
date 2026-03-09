#!/usr/bin/env bash
# =============================================================================
# Wisper Genie — Installer
# Installs the AI dictation tool for macOS.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/OWNER/wisper-genie/main/install.sh | bash
#
# Safe to run multiple times (idempotent).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_URL="https://github.com/glp942001/wisper-genie.git"   # HTTPS (tried first)
REPO_SSH="git@github.com:glp942001/wisper-genie.git"      # SSH   (fallback)
INSTALL_DIR="$HOME/.wisper-genie"
VENV_DIR="$INSTALL_DIR/.venv"
WRAPPER_SRC="$INSTALL_DIR/scripts/dictation-wrapper"
WRAPPER_DST="$HOME/.local/bin/wisper-genie"
MODEL_DIR="$INSTALL_DIR/models"
WHISPER_MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
WHISPER_MODEL_FILE="$MODEL_DIR/ggml-small.bin"
OLLAMA_MODEL="qwen3.5:2b"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=12
TOTAL_STEPS=12

# ---------------------------------------------------------------------------
# Colors & helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

_step=0

step() {
    _step=$(( _step + 1 ))
    printf "\n${BLUE}${BOLD}[%d/%d]${RESET} %s\n" "$_step" "$TOTAL_STEPS" "$1"
}

ok() {
    printf "  ${GREEN}%s${RESET} %s\n" "✔" "$1"
}

warn() {
    printf "  ${YELLOW}%s${RESET} %s\n" "⚠" "$1"
}

fail() {
    printf "  ${RED}%s${RESET} %s\n" "✘" "$1" >&2
    exit 1
}

info() {
    printf "  ${BOLD}→${RESET} %s\n" "$1"
}

clear_quarantine() {
    # macOS Gatekeeper adds com.apple.quarantine to downloaded files, causing
    # "unidentified developer" or "damaged" warnings. Strip it silently.
    local target="$1"
    if xattr -l "$target" 2>/dev/null | grep -q "com.apple.quarantine"; then
        xattr -dr com.apple.quarantine "$target" 2>/dev/null || true
    fi
}

# ---------------------------------------------------------------------------
# Require macOS
# ---------------------------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "Wisper Genie requires macOS. Detected: $(uname -s)"
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
printf "\n"
printf "${BOLD}========================================${RESET}\n"
printf "${BOLD}   Wisper Genie — Installer${RESET}\n"
printf "${BOLD}========================================${RESET}\n"
printf "  AI-powered dictation for macOS\n"
printf "\n"

# ===========================================================================
# Step 1: Homebrew
# ===========================================================================
step "Checking for Homebrew..."

install_homebrew() {
    info "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # After install, make sure brew is on PATH for Apple Silicon & Intel
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -f /usr/local/bin/brew ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
}

if command -v brew &>/dev/null; then
    ok "Homebrew is installed ($(brew --version | head -1))"
else
    install_homebrew
    if command -v brew &>/dev/null; then
        ok "Homebrew installed successfully"
    else
        fail "Could not install Homebrew. Please install manually: https://brew.sh"
    fi
fi

# ===========================================================================
# Step 2: Python 3.12+
# ===========================================================================
step "Checking for Python 3.12+..."

python_is_sufficient() {
    local py="$1"
    if ! command -v "$py" &>/dev/null; then
        return 1
    fi
    local version
    version=$("$py" --version 2>&1 | awk '{print $2}')
    local major minor
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    [[ "$major" -gt "$MIN_PYTHON_MAJOR" ]] && return 0
    [[ "$major" -eq "$MIN_PYTHON_MAJOR" && "$minor" -ge "$MIN_PYTHON_MINOR" ]] && return 0
    return 1
}

# Try to find a suitable Python
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3; do
    if python_is_sufficient "$candidate"; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [[ -n "$PYTHON_BIN" ]]; then
    ok "Found $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
else
    info "Python 3.12+ not found. Installing python@3.13 via Homebrew..."
    brew install python@3.13
    # Refresh PATH so the new python is visible
    eval "$(brew shellenv 2>/dev/null || true)"
    hash -r 2>/dev/null || true

    for candidate in python3.13 python3.12 python3; do
        if python_is_sufficient "$candidate"; then
            PYTHON_BIN="$candidate"
            break
        fi
    done

    if [[ -z "$PYTHON_BIN" ]]; then
        # Try the Homebrew-specific path directly
        if python_is_sufficient "$(brew --prefix)/bin/python3.13"; then
            PYTHON_BIN="$(brew --prefix)/bin/python3.13"
        else
            fail "Failed to install Python 3.12+. Please install manually: brew install python@3.13"
        fi
    fi
    ok "Installed $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
fi

# ===========================================================================
# Step 3: Git
# ===========================================================================
step "Checking for Git..."

if command -v git &>/dev/null; then
    ok "Git is installed ($(git --version))"
else
    info "Git not found. Installing via Homebrew..."
    brew install git
    if command -v git &>/dev/null; then
        ok "Git installed successfully"
    else
        fail "Could not install Git. Please install manually: brew install git"
    fi
fi

# ===========================================================================
# Step 4: Clone / update repository
# ===========================================================================
step "Setting up Wisper Genie repository..."

clone_repo() {
    info "Cloning repository..."
    # Try HTTPS first, fall back to SSH
    if git clone "$REPO_URL" "$INSTALL_DIR" 2>/dev/null; then
        ok "Cloned via HTTPS"
    elif git clone "$REPO_SSH" "$INSTALL_DIR" 2>/dev/null; then
        ok "Cloned via SSH"
    else
        fail "Could not clone repository. Make sure you have access to the private repo.\n  HTTPS: $REPO_URL\n  SSH:   $REPO_SSH\n  Configure SSH keys or a personal access token first."
    fi
    # Strip macOS quarantine flags from all repo files
    clear_quarantine "$INSTALL_DIR"
}

if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Repository already exists. Pulling latest changes..."
    if git -C "$INSTALL_DIR" pull --ff-only 2>/dev/null; then
        ok "Updated to latest version"
    else
        warn "Could not fast-forward. Attempting reset to remote..."
        local_branch=$(git -C "$INSTALL_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
        remote_branch="origin/$local_branch"
        if git -C "$INSTALL_DIR" fetch origin && git -C "$INSTALL_DIR" reset --hard "$remote_branch" 2>/dev/null; then
            ok "Reset to latest remote version"
        else
            warn "Pull failed — continuing with existing version"
        fi
    fi
elif [[ -d "$INSTALL_DIR" ]]; then
    # Directory exists but is not a git repo — back up and re-clone
    warn "$INSTALL_DIR exists but is not a git repository. Backing up..."
    backup_dir="${INSTALL_DIR}.backup.$(date +%s)"
    mv "$INSTALL_DIR" "$backup_dir"
    info "Backup saved to $backup_dir"
    clone_repo
else
    clone_repo
fi

# ===========================================================================
# Step 5: Python virtual environment & dependencies
# ===========================================================================
step "Creating Python virtual environment & installing dependencies..."

if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists. Upgrading pip & reinstalling..."
else
    info "Creating venv at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade pip silently
pip install --upgrade pip --quiet 2>/dev/null

# Install the project in editable mode
info "Installing project dependencies (this may take a minute)..."
if pip install -e "$INSTALL_DIR[dev]" --quiet 2>/dev/null; then
    ok "Dependencies installed"
else
    # Retry without --quiet so the user can see the error
    warn "Retrying install with verbose output..."
    pip install -e "$INSTALL_DIR[dev]"
    ok "Dependencies installed"
fi

# Strip quarantine from venv binaries (pip downloads can be flagged)
clear_quarantine "$VENV_DIR"

deactivate 2>/dev/null || true

# ===========================================================================
# Step 6: Install wrapper script
# ===========================================================================
step "Installing wisper-genie command..."

mkdir -p "$(dirname "$WRAPPER_DST")"

if [[ -f "$WRAPPER_SRC" ]]; then
    cp "$WRAPPER_SRC" "$WRAPPER_DST"
    chmod +x "$WRAPPER_DST"
    clear_quarantine "$WRAPPER_DST"
    ok "Installed wrapper to $WRAPPER_DST"
else
    fail "Wrapper script not found at $WRAPPER_SRC. The repository may be incomplete."
fi

# ===========================================================================
# Step 7: Ensure ~/.local/bin is on PATH
# ===========================================================================
step "Checking PATH..."

if echo "$PATH" | tr ':' '\n' | grep -qx "$HOME/.local/bin"; then
    ok "\$HOME/.local/bin is already in PATH"
else
    SHELL_RC="$HOME/.zshrc"
    EXPORT_LINE='export PATH="$HOME/.local/bin:$PATH"'

    # Also check .zshrc content in case it's there but not yet sourced
    if [[ -f "$SHELL_RC" ]] && grep -qF '.local/bin' "$SHELL_RC"; then
        warn "\$HOME/.local/bin found in $SHELL_RC but not in current PATH"
        info "It will be available in new terminal sessions"
    else
        info "Adding \$HOME/.local/bin to PATH in $SHELL_RC..."
        echo "" >> "$SHELL_RC"
        echo "# Added by Wisper Genie installer" >> "$SHELL_RC"
        echo "$EXPORT_LINE" >> "$SHELL_RC"
        ok "Updated $SHELL_RC"
        info "Run 'source ~/.zshrc' or open a new terminal to use the 'wisper-genie' command"
    fi

    # Also export for the remainder of this script
    export PATH="$HOME/.local/bin:$PATH"
fi

# ===========================================================================
# Step 8: Ollama (optional)
# ===========================================================================
step "Ollama setup (AI text cleanup)..."

install_ollama=false
if command -v ollama &>/dev/null; then
    ok "Ollama is already installed"
    install_ollama=false
else
    # When running piped from curl, stdin is the script, so re-open /dev/tty
    if [[ -t 0 ]]; then
        read_input() { read "$@"; }
    else
        read_input() { read "$@" < /dev/tty; }
    fi

    printf "  Ollama is required for AI text cleanup. Install now? ${BOLD}[Y/n]:${RESET} "
    if read_input -r answer 2>/dev/null; then
        case "${answer:-Y}" in
            [Yy]*|"")
                install_ollama=true
                ;;
            *)
                warn "Skipping Ollama. You can install later: brew install --cask ollama"
                ;;
        esac
    else
        # Non-interactive — default to yes
        info "Non-interactive mode detected. Installing Ollama..."
        install_ollama=true
    fi
fi

if [[ "$install_ollama" == true ]]; then
    info "Installing Ollama..."
    brew install --cask ollama
    if command -v ollama &>/dev/null || [[ -d "/Applications/Ollama.app" ]]; then
        ok "Ollama installed"
        # Remove Gatekeeper quarantine so Ollama opens without "unidentified developer" warning
        if [[ -d "/Applications/Ollama.app" ]]; then
            clear_quarantine "/Applications/Ollama.app"
        fi
        info "Starting Ollama.app..."
        open -a Ollama 2>/dev/null || true
        # Give Ollama a moment to start its server
        sleep 3
    else
        warn "Ollama installation may have failed. Install manually: brew install --cask ollama"
    fi
fi

# ===========================================================================
# Step 9: Download Whisper model
# ===========================================================================
step "Downloading Whisper model (small)..."

mkdir -p "$MODEL_DIR"

if [[ -f "$WHISPER_MODEL_FILE" ]]; then
    file_size=$(stat -f%z "$WHISPER_MODEL_FILE" 2>/dev/null || echo "0")
    # The small model is ~465 MB; skip if file looks complete (> 400 MB)
    if [[ "$file_size" -gt 400000000 ]]; then
        ok "Whisper small model already downloaded ($(( file_size / 1048576 )) MB)"
    else
        warn "Existing model file looks incomplete ($(( file_size / 1048576 )) MB). Re-downloading..."
        rm -f "$WHISPER_MODEL_FILE"
        info "Downloading ggml-small.bin (~465 MB) — this may take a few minutes..."
        curl -L --progress-bar "$WHISPER_MODEL_URL" -o "$WHISPER_MODEL_FILE"
        clear_quarantine "$WHISPER_MODEL_FILE"
        ok "Whisper small model downloaded"
    fi
else
    info "Downloading ggml-small.bin (~465 MB) — this may take a few minutes..."
    curl -L --progress-bar "$WHISPER_MODEL_URL" -o "$WHISPER_MODEL_FILE"
    clear_quarantine "$WHISPER_MODEL_FILE"
    ok "Whisper small model downloaded"
fi

# ===========================================================================
# Step 10: Pull Ollama model
# ===========================================================================
step "Pulling Ollama model ($OLLAMA_MODEL)..."

if command -v ollama &>/dev/null; then
    # Check if the model is already pulled
    if ollama list 2>/dev/null | grep -q "qwen3.5"; then
        ok "Model $OLLAMA_MODEL is already available"
    else
        info "Pulling $OLLAMA_MODEL — this may take a minute..."
        if ollama pull "$OLLAMA_MODEL"; then
            ok "Model $OLLAMA_MODEL pulled successfully"
        else
            warn "Could not pull $OLLAMA_MODEL. Make sure Ollama is running, then run: ollama pull $OLLAMA_MODEL"
        fi
    fi
else
    warn "Ollama not installed — skipping model pull"
    info "Install Ollama later, then run: ollama pull $OLLAMA_MODEL"
fi

# ===========================================================================
# Step 11: macOS permissions
# ===========================================================================
step "Setting up macOS permissions..."

# Detect which terminal app is running (for clearer instructions)
TERM_APP="your terminal app"
if [[ "$TERM_PROGRAM" == "Apple_Terminal" ]]; then
    TERM_APP="Terminal"
elif [[ "$TERM_PROGRAM" == "iTerm.app" ]]; then
    TERM_APP="iTerm2"
elif [[ "$TERM_PROGRAM" == "WarpTerminal" ]]; then
    TERM_APP="Warp"
elif [[ -n "${TERM_PROGRAM:-}" ]]; then
    TERM_APP="$TERM_PROGRAM"
fi

printf "\n"
printf "  ${YELLOW}${BOLD}Wisper Genie needs two macOS permissions to work:${RESET}\n"
printf "\n"
printf "  ${BOLD}1. Accessibility${RESET} — so it can read screen context and paste text\n"
printf "  ${BOLD}2. Microphone${RESET}    — so it can hear you speak\n"
printf "\n"
printf "  Both require adding ${BOLD}${TERM_APP}${RESET} in System Settings.\n"
printf "\n"

# --- Accessibility ---
printf "  ${BLUE}${BOLD}Opening Accessibility settings...${RESET}\n"
printf "  → Add ${BOLD}${TERM_APP}${RESET} to the list and toggle it ON.\n"
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility" 2>/dev/null || \
    open "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_Accessibility" 2>/dev/null || true

printf "  Press ${BOLD}Enter${RESET} once you've granted Accessibility access..."
if [[ -t 0 ]]; then
    read -r
else
    read -r < /dev/tty 2>/dev/null || true
fi
ok "Accessibility — done"

# --- Microphone ---
printf "\n"
printf "  ${BLUE}${BOLD}Opening Microphone settings...${RESET}\n"
printf "  → Toggle ON microphone access for ${BOLD}${TERM_APP}${RESET}.\n"
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone" 2>/dev/null || \
    open "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_Microphone" 2>/dev/null || true

printf "  Press ${BOLD}Enter${RESET} once you've granted Microphone access..."
if [[ -t 0 ]]; then
    read -r
else
    read -r < /dev/tty 2>/dev/null || true
fi
ok "Microphone — done"

# ===========================================================================
# Step 12: Set up hotkey
# ===========================================================================
_step=$(( _step + 1 ))
printf "\n${BLUE}${BOLD}[%d/%d]${RESET} %s\n" "$_step" "$TOTAL_STEPS" "Setting up your push-to-talk key..."

CONFIG_FILE="$INSTALL_DIR/config/default.toml"

printf "\n"
printf "  Press the key you want to use for push-to-talk.\n"
printf "  ${BOLD}Recommended: Right Option or Right Control.${RESET}\n"
printf "\n"
printf "  Press your chosen key now...\n"

# Detect whatever key the user presses
DETECTED_KEY=$("$VENV_DIR/bin/python" -c "
import time
from pynput import keyboard
result = [None]
def on_press(key):
    name = getattr(key, 'name', None)
    if name:
        result[0] = name
        return False
listener = keyboard.Listener(on_press=on_press)
listener.start()
for _ in range(150):
    time.sleep(0.1)
    if result[0]:
        break
listener.stop()
print(result[0] or 'TIMEOUT')
" 2>/dev/null)

if [[ "$DETECTED_KEY" == "TIMEOUT" || -z "$DETECTED_KEY" ]]; then
    warn "No key detected. Using Right Option as default."
    info "Change later in ~/.wisper-genie/config/default.toml"
else
    ok "Detected: $DETECTED_KEY"

    # Write the key to config
    "$VENV_DIR/bin/python" -c "
import sys
key = sys.argv[1]
config = open('$CONFIG_FILE').read()
import re
config = re.sub(r'keys = \[\"[^\"]*\"\]', f'keys = [\"<{key}>\"]', config)
open('$CONFIG_FILE', 'w').write(config)
" "$DETECTED_KEY" 2>/dev/null

    ok "Hotkey set to: $DETECTED_KEY"
    printf "\n"
    printf "  ${BOLD}Verify:${RESET} Press ${BOLD}$DETECTED_KEY${RESET} one more time to confirm...\n"

    # Verify
    VERIFY_KEY=$("$VENV_DIR/bin/python" -c "
import time
from pynput import keyboard
result = [None]
def on_press(key):
    name = getattr(key, 'name', None)
    if name:
        result[0] = name
        return False
listener = keyboard.Listener(on_press=on_press)
listener.start()
for _ in range(100):
    time.sleep(0.1)
    if result[0]:
        break
listener.stop()
print(result[0] or 'TIMEOUT')
" 2>/dev/null)

    if [[ "$VERIFY_KEY" == "$DETECTED_KEY" ]]; then
        ok "Confirmed! Push-to-talk key: $DETECTED_KEY"
    elif [[ "$VERIFY_KEY" != "TIMEOUT" ]]; then
        warn "Got '$VERIFY_KEY' instead of '$DETECTED_KEY'. Using '$VERIFY_KEY'."
        "$VENV_DIR/bin/python" -c "
import sys, re
key = sys.argv[1]
config = open('$CONFIG_FILE').read()
config = re.sub(r'keys = \[\"[^\"]*\"\]', f'keys = [\"<{key}>\"]', config)
open('$CONFIG_FILE', 'w').write(config)
" "$VERIFY_KEY" 2>/dev/null
        ok "Hotkey updated to: $VERIFY_KEY"
    else
        info "No confirmation received — keeping $DETECTED_KEY."
    fi
fi

# ===========================================================================
# Done!
# ===========================================================================
printf "\n"
printf "${GREEN}${BOLD}========================================${RESET}\n"
printf "${GREEN}${BOLD}   Installation complete!${RESET}\n"
printf "${GREEN}${BOLD}========================================${RESET}\n"
printf "\n"
printf "${BOLD}How to use:${RESET}\n"
printf "\n"
printf "  ${BOLD}Step 1:${RESET} Open a terminal and run:\n"
printf "         ${BOLD}wisper-genie${RESET}\n"
printf "\n"
printf "  ${BOLD}Step 2:${RESET} Switch to any app (Slack, Gmail, Google, Notes, etc.)\n"
printf "\n"
printf "  ${BOLD}Step 3:${RESET} Click on a text field, then:\n"
printf "         ${BOLD}Hold Right Option key${RESET} → you'll hear a sound → speak naturally\n"
printf "         ${BOLD}Release Right Option key${RESET} → your text appears, formatted and clean\n"
printf "\n"
printf "  That's it! Each press is independent — switch apps freely.\n"
printf "\n"
printf "${BOLD}Other commands:${RESET}\n"
printf "  ${BOLD}wisper-genie install${RESET}      Re-download models\n"
printf "  ${BOLD}wisper-genie autostart${RESET}    Launch on login\n"
printf "  ${BOLD}wisper-genie uninstall${RESET}    Remove from your system\n"
printf "  ${BOLD}wisper-genie --help${RESET}       Show all commands\n"
printf "\n"
if ! echo "$PATH" | tr ':' '\n' | grep -qx "$HOME/.local/bin"; then
    printf "${YELLOW}Note:${RESET} Run ${BOLD}source ~/.zshrc${RESET} or open a new terminal before using 'wisper-genie'.\n"
    printf "\n"
fi
printf "Happy dictating! 🎙️\n\n"
