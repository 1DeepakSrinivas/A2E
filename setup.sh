#!/usr/bin/env bash
# Set up this machine: packages, Rust, llama.cpp, the a2e binary, models, sandbox and checks.
#   ./setup.sh [--skip-llama] [--skip-models]
#   ./setup.sh telemetry      allow powermetrics without a password (macOS only)
# Runs on 64-bit Raspberry Pi OS and Apple Silicon macOS. Every step can be repeated.
# This stays a shell script because the a2e program does not exist yet when it first runs.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

usage() { echo "usage: ./setup.sh [--skip-llama] [--skip-models] | telemetry" >&2; exit 2; }
step() { printf '\n==> %s\n' "$1"; }
die() { echo "$1" >&2; exit 2; }

# Pins come from the repository files, so they are written in one place.
rust_version() {
    local v
    v="$(sed -n 's/^rust-version = "\(.*\)"$/\1/p' Cargo.toml)"
    [[ "$v" == *.*.* ]] || v="$v.0"
    echo "$v"
}
llama_pin() { sed -n 's/^commit = "\([0-9a-f]*\)"$/\1/p' configs/models.toml; }

detect_platform() {
    if [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] && command -v apt-get >/dev/null &&
       [[ "$(dpkg --print-architecture)" == arm64 ]]; then
        platform=pi
        jobs="${A2E_BUILD_JOBS:-1}"
    elif [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
        platform=mac
        jobs="${A2E_BUILD_JOBS:-$(sysctl -n hw.ncpu)}"
    else
        die "Use 64-bit Raspberry Pi OS or an Apple Silicon Mac."
    fi
}

install_deps() {
    case "$platform" in
        pi)
            sudo apt-get update
            sudo apt-get install -y build-essential cmake git curl ca-certificates tmux libraspberrypi-bin
            if ! id -nG | tr ' ' '\n' | grep -qx video; then
                sudo usermod -aG video "$USER"
                echo "Added $USER to the video group. Log out and back in before measuring."
            fi
            ;;
        mac)
            xcode-select -p >/dev/null 2>&1 || die "Install the Xcode command line tools: xcode-select --install"
            command -v brew >/dev/null || die "Install Homebrew (https://brew.sh), then run setup.sh again."
            for pkg in cmake tmux; do
                brew list --formula "$pkg" >/dev/null 2>&1 || brew install "$pkg"
            done
            ;;
    esac
    if ! command -v rustup >/dev/null; then
        curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs |
            sh -s -- -y --profile minimal --default-toolchain none
    fi
    rustup toolchain install "$rust" --profile minimal
}

# Builds llama-server at the pinned commit and checks that its version string carries it.
build_llama() {
    local commit="${LLAMA_COMMIT:-$(llama_pin)}"
    local deps_dir="${A2E_HOME:-$PWD/deps}"
    local source_dir="$deps_dir/llama.cpp" metal=OFF version
    local build_dir="$source_dir/build-a2e"
    if [[ ! "$commit" =~ ^[a-f0-9]{40}$ || ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
        die "Use a full llama.cpp commit SHA and a positive A2E_BUILD_JOBS value."
    fi
    mkdir -p "$deps_dir"
    if [[ ! -d "$source_dir" ]]; then
        git clone --no-checkout --depth 1 https://github.com/ggml-org/llama.cpp.git "$source_dir"
    fi
    if ! git -C "$source_dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
        git -C "$source_dir" fetch --depth 1 origin "$commit"
    fi
    if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$commit" || ! -f "$source_dir/CMakeLists.txt" ]]; then
        git -C "$source_dir" checkout --detach "$commit"
    fi
    # The runner looks for the first 8 characters of the pin in the version string.
    git -C "$source_dir" config core.abbrev 8
    [[ "$platform" == mac ]] && metal=ON
    cmake -S "$source_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CCACHE=OFF -DGGML_NATIVE=ON -DGGML_METAL="$metal" -DLLAMA_BUILD_TESTS=OFF
    cmake --build "$build_dir" --parallel "$jobs" --target llama-server llama-bench
    version="$("$build_dir/bin/llama-server" --version 2>&1)"
    printf '%s\n' "$version"
    if [[ "$version" != *"${commit:0:8}"* ]]; then
        echo "llama-server --version does not contain ${commit:0:8}; the runner would reject it." >&2
        exit 1
    fi
}

build_a2e() {
    # Link-time optimization needs more memory than the 1 GB Pi has.
    [[ "$platform" == pi ]] && export CARGO_PROFILE_RELEASE_LTO=false
    cargo "+$rust" build --release --locked --jobs "$jobs"
}

post_checks() {
    local bin=target/release/a2e
    "$bin" tools validate
    "$bin" cases validate
    if [[ "$platform" == mac ]]; then "$bin" doctor --probe; else "$bin" doctor; fi
    if [[ "$platform" == pi ]]; then
        if [[ -n "$(swapon --show --noheadings 2>/dev/null)" ]]; then
            echo "Swap is on. The Pi baseline refuses to run with swap on: sudo dphys-swapfile swapoff && sudo systemctl disable dphys-swapfile"
        fi
        if command -v vcgencmd >/dev/null; then
            echo "throttled: $(vcgencmd get_throttled)  (0x0 is clean)"
        fi
        if [[ ! -f .env ]]; then
            echo "No .env. Results stay in results/outbox until you copy .env.example to .env and set WORKER_URL and WRITE_TOKEN."
        fi
    fi
}

# Lets the runner call powermetrics through sudo without a password. Undo with:
#   sudo rm /etc/sudoers.d/a2e-powermetrics
mac_telemetry() {
    [[ "$(uname -s)" == Darwin ]] || die "This step is for macOS only."
    [[ "$(id -u)" -ne 0 ]] || die "Run this as your normal user, not with sudo."
    local rule_file=/etc/sudoers.d/a2e-powermetrics tmp
    tmp="$(mktemp)"
    trap 'rm -f "$tmp"' EXIT
    # Keep these arguments in sync with src/measure.rs.
    local sampler=(/usr/bin/powermetrics --samplers "cpu_power,gpu_power,thermal" -i 500)
    local sampler_rule="${sampler[*]}"
    printf '%s ALL=(root) NOPASSWD: %s\n' "$(id -un)" "${sampler_rule//,/\\,}" > "$tmp"
    sudo visudo -cf "$tmp"
    sudo install -m 0440 -o root -g wheel "$tmp" "$rule_file"
    sudo -n -l -- "${sampler[@]}" > /dev/null
    echo "Installed $rule_file. Check with: target/release/a2e doctor --probe"
}

if [[ "${1:-}" == telemetry ]]; then mac_telemetry; exit; fi

skip_llama=0
skip_models=0
for arg in "$@"; do
    case "$arg" in
        --skip-llama) skip_llama=1 ;;
        --skip-models) skip_models=1 ;;
        *) usage ;;
    esac
done

[[ "$(id -u)" != 0 ]] || die "Run setup.sh as your normal user. Package installation uses sudo."
detect_platform
rust="${A2E_RUST:-$(rust_version)}"
export PATH="$HOME/.cargo/bin:$PATH"

step "Packages and Rust $rust"
install_deps

if [[ "$skip_llama" == 0 ]]; then
    step "llama.cpp at the pinned commit"
    build_llama
fi

step "a2e binary"
build_a2e

if [[ "$skip_models" == 0 ]]; then
    step "Models"
    target/release/a2e models fetch
fi

step "File sandbox"
target/release/a2e sandbox init
target/release/a2e sandbox verify

step "Checks"
post_checks

printf '\nSetup done. Open the menu with:\n  target/release/a2e\nor start a suite directly:\n  target/release/a2e start smoke\n'
