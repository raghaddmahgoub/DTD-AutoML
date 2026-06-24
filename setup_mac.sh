#!/usr/bin/env bash
# setup_mac.sh — one-shot macOS setup for the GP AutoML project
#
# Usage (from repo root):
#   chmod +x setup_mac.sh
#   ./setup_mac.sh
#
# Options (environment variables):
#   VENV_DIR=./automl_env_310   — virtualenv location (default)
#   PYTHON=python3.11           — force a specific Python binary
#   SKIP_DATASETS=1             — skip downloading sample CSVs
#   SKIP_VERIFY=1               — skip import smoke tests
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/automl_env_310}"
REQ_FILE="$REPO_ROOT/requirements-mac.txt"
PYTHON_BIN=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[setup]${NC} $*"; }
ok()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
fail()  { echo -e "${RED}[setup]${NC} $*" >&2; exit 1; }

is_yes() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    y|yes) return 0 ;;
    *) return 1 ;;
  esac
}

header() {
  echo ""
  echo "============================================================"
  echo " GP AutoML — macOS setup"
  echo " Repo: $REPO_ROOT"
  echo "============================================================"
}

# ---------------------------------------------------------------------------
# 1. macOS + architecture
# ---------------------------------------------------------------------------
check_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    warn "This script is tuned for macOS. Continuing anyway..."
    return
  fi
  local arch
  arch="$(uname -m)"
  if [[ "$arch" == "arm64" ]]; then
    ok "Apple Silicon detected ($arch)"
  else
    ok "Intel Mac detected ($arch)"
  fi
}

# ---------------------------------------------------------------------------
# 2. Homebrew system libraries (optional but recommended)
# ---------------------------------------------------------------------------
check_brew_deps() {
  if ! command -v brew >/dev/null 2>&1; then
    warn "Homebrew not found. Install from https://brew.sh for best results."
    warn "Some packages (XGBoost/LightGBM) work better with: brew install libomp"
    return
  fi

  local missing=()
  for pkg in libomp graphviz; do
    if ! brew list "$pkg" &>/dev/null; then
      missing+=("$pkg")
    fi
  done

  if ((${#missing[@]} > 0)); then
    warn "Recommended Homebrew packages missing: ${missing[*]}"
    if [[ "${NONINTERACTIVE:-0}" == "1" ]]; then
      warn "NONINTERACTIVE=1 — skipping brew install"
    else
      read -r -p "Install via Homebrew now? [y/N] " ans
      if is_yes "$ans"; then
        brew install "${missing[@]}"
        ok "Installed: ${missing[*]}"
      else
        warn "Skipped. You can install later: brew install libomp graphviz"
      fi
    fi
  else
    ok "Homebrew deps present (libomp, graphviz)"
  fi
}

# ---------------------------------------------------------------------------
# 3. Python 3.10–3.12 (reject 3.13+)
# ---------------------------------------------------------------------------
python_version_ok() {
  local ver="$1"
  local major minor
  major="$(echo "$ver" | cut -d. -f1)"
  minor="$(echo "$ver" | cut -d. -f2)"
  [[ "$major" -eq 3 && "$minor" -ge 10 && "$minor" -le 12 ]]
}

find_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "$PYTHON"
    return
  fi

  local candidates=(
    python3.12 python3.11 python3.10
    /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10
    /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.10
    python3
  )

  for bin in "${candidates[@]}"; do
  if command -v "$bin" >/dev/null 2>&1; then
    local ver
    ver="$("$bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [[ -n "$ver" ]] && python_version_ok "$ver"; then
      echo "$bin"
      return
    fi
  fi
  done
  echo ""
}

setup_venv() {
  local py_bin="$1"
  local ver
  ver="$("$py_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"

  if ! python_version_ok "$(echo "$ver" | cut -d. -f1-2)"; then
    fail "Need Python 3.10–3.12. Found $py_bin ($ver). Python 3.13+ breaks some deps (LangChain/Pydantic)."
  fi
  ok "Using $py_bin (Python $ver)"

  if [[ -d "$VENV_DIR" ]]; then
    warn "Virtualenv already exists: $VENV_DIR"
    if [[ "${NONINTERACTIVE:-0}" == "1" ]]; then
      info "NONINTERACTIVE=1 — keeping existing venv"
    else
      read -r -p "Recreate it? This deletes the old env. [y/N] " ans
      if is_yes "$ans"; then
        rm -rf "$VENV_DIR"
      fi
    fi
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtualenv at $VENV_DIR"
    "$py_bin" -m venv "$VENV_DIR"
  fi

  PYTHON_BIN="$VENV_DIR/bin/python"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    fail "Venv python not found at $PYTHON_BIN"
  fi
  ok "Using venv python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"
}

install_packages() {
  info "Upgrading pip..."
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

  if [[ ! -f "$REQ_FILE" ]]; then
    fail "Missing $REQ_FILE"
  fi

  info "Installing Python packages (this may take several minutes)..."
  info "  • Core ML stack (numpy, pandas, sklearn, xgboost, lightgbm)"
  info "  • AutoGluon tabular"
  info "  • LangChain / LangGraph"
  info "  • Dask, Optuna, OpenML"

  # Stage installs — helps when wheels fail mid-way
  "$PYTHON_BIN" -m pip install "numpy>=1.26.0,<2.2" "pandas>=2.1.0" "scipy>=1.11.0"
  "$PYTHON_BIN" -m pip install -r "$REQ_FILE"

  ok "Python packages installed"
}

setup_env_file() {
  if [[ -f "$REPO_ROOT/.env" ]]; then
    ok ".env already exists"
    return
  fi
  if [[ -f "$REPO_ROOT/.env.example" ]]; then
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    warn "Created .env from .env.example — add your GOOGLE_API_KEY before running LLM steps"
  else
    cat > "$REPO_ROOT/.env" <<'EOF'
GOOGLE_API_KEY=your_gemini_api_key_here
EOF
    warn "Created .env — add your GOOGLE_API_KEY"
  fi
}

download_datasets() {
  if [[ "${SKIP_DATASETS:-0}" == "1" ]]; then
    info "Skipping sample dataset download (SKIP_DATASETS=1)"
    return
  fi

  info "Downloading / creating sample datasets..."
  "$PYTHON_BIN" <<PY
from pathlib import Path
root = Path("${REPO_ROOT}")
cls_dir = root / "assets" / "data" / "Datasets" / "Classification Datasets"
reg_dir = root / "assets" / "data" / "Datasets" / "Regression Datasets"
cls_dir.mkdir(parents=True, exist_ok=True)
reg_dir.mkdir(parents=True, exist_ok=True)

def save(df, path):
    if not path.exists():
        df.to_csv(path, index=False)
        print(f"  created {path}")
    else:
        print(f"  exists  {path}")

from sklearn.datasets import load_iris, load_wine, load_diabetes, fetch_california_housing

iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={"target": "species"})
save(df, cls_dir / "Iris.csv")

wine = load_wine(as_frame=True)
save(wine.frame, cls_dir / "wine.csv")

diabetes = load_diabetes(as_frame=True)
save(diabetes.frame, reg_dir / "diabetes.csv")

housing = fetch_california_housing(as_frame=True)
save(housing.frame.rename(columns={"MedHouseVal": "median_house_value"}), reg_dir / "California Housing Prices.csv")

fallback = root / "output" / "test_pipeline" / "iris_sample.csv"
fallback.parent.mkdir(parents=True, exist_ok=True)
if not fallback.exists():
    df.to_csv(fallback, index=False)
    print(f"  created {fallback}")
PY
  ok "Sample datasets ready under assets/data/Datasets/"
}

verify_install() {
  if [[ "${SKIP_VERIFY:-0}" == "1" ]]; then
    return
  fi

  info "Verifying imports..."
  "$PYTHON_BIN" <<'PY'
import sys
errors = []

def check(label, fn):
    try:
        fn()
        print(f"  OK  {label}")
    except Exception as exc:
        print(f"  FAIL {label}: {exc}")
        errors.append(label)

check("numpy", lambda: __import__("numpy"))
check("pandas", lambda: __import__("pandas"))
check("sklearn", lambda: __import__("sklearn"))
check("optuna", lambda: __import__("optuna"))
check("xgboost", lambda: __import__("xgboost"))
check("lightgbm", lambda: __import__("lightgbm"))
check("dask", lambda: __import__("dask"))
check("langchain_core", lambda: __import__("langchain_core"))
check("langgraph", lambda: __import__("langgraph"))
check("langchain_google_genai", lambda: __import__("langchain_google_genai"))
check("openml", lambda: __import__("openml"))
check("autogluon.tabular", lambda: __import__("autogluon.tabular"))
check("AutoGluon TabularPredictor", lambda: __import__("autogluon.tabular", fromlist=["TabularPredictor"]))
check("torch (NN_TORCH)", lambda: __import__("torch"))
check("fastai (FASTAI)", lambda: __import__("fastai"))

if errors:
    print("\nSome packages failed:", ", ".join(errors))
    sys.exit(1)
print("\nAll core imports passed.")
PY
  ok "Verification passed"
}

print_next_steps() {
  echo ""
  echo "============================================================"
  echo -e "${GREEN}Setup complete!${NC}"
  echo "============================================================"
  echo ""
  echo "1. Activate the environment:"
  echo "     source $VENV_DIR/bin/activate"
  echo "     # or run directly: $PYTHON_BIN"
  echo ""
  echo "2. Add your Gemini API key to .env:"
  echo "     GOOGLE_API_KEY=..."
  echo ""
  echo "3. Test the training pipeline (LLM picks models, you approve):"
  echo "     python src/test_tools_pipeline.py --mode manual --data iris"
  echo ""
  echo "4. Non-interactive quick test:"
  echo "     python src/test_tools_pipeline.py --mode manual --no-prompts"
  echo ""
  echo "5. OpenML benchmark:"
  echo "     python benchmark/openml_benchmark.py"
  echo ""
  echo "Tip: always use 'python' from the venv, not system 'python3'."
  echo "============================================================"
}

# ---------------------------------------------------------------------------
# Run setup
# ---------------------------------------------------------------------------
header
check_macos
check_brew_deps

PY_BIN="$(find_python)"
if [[ -z "$PY_BIN" ]]; then
  fail "No suitable Python 3.10–3.12 found.
Install one with Homebrew:
  brew install python@3.11
Then re-run:
  PYTHON=python3.11 ./setup_mac.sh"
fi

setup_venv "$PY_BIN"
install_packages
setup_env_file
download_datasets
verify_install
print_next_steps
