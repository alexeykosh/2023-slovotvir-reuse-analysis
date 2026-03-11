#!/bin/bash

# Setup script for Slovotvir Reuse Analysis project
# This script creates a virtual environment and installs all dependencies
# Requires Python 3.10 or 3.11 (tensorflow/bayesflow do not support newer versions)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Slovotvir Reuse Analysis environment...${NC}"

# Function to check Python version compatibility
check_python_compat() {
    local python_cmd=$1
    local version=$($python_cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    if [[ "$version" == "3.10" ]] || [[ "$version" == "3.11" ]]; then
        echo "$version"
        return 0
    fi
    return 1
}

# Try to find a compatible Python version
PYTHON_CMD=""

# Check python3.11 first
if command -v python3.11 &> /dev/null; then
    if check_python_compat python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        PYTHON_VERSION=$(check_python_compat python3.11)
    fi
fi

# Check python3.10
if [[ -z "$PYTHON_CMD" ]] && command -v python3.10 &> /dev/null; then
    if check_python_compat python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
        PYTHON_VERSION=$(check_python_compat python3.10)
    fi
fi

# Check default python3
if [[ -z "$PYTHON_CMD" ]] && command -v python3 &> /dev/null; then
    if check_python_compat python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(check_python_compat python3)
    fi
fi

# Check pyenv versions
if [[ -z "$PYTHON_CMD" ]] && command -v pyenv &> /dev/null; then
    for ver in 3.11 3.10; do
        if pyenv versions --bare | grep -q "^$ver"; then
            PYENV_VERSION=$(pyenv versions --bare | grep "^$ver" | tail -1)
            echo -e "${YELLOW}Found Python $PYENV_VERSION via pyenv${NC}"
            echo -e "${YELLOW}Run: pyenv local $PYENV_VERSION${NC}"
            echo -e "${YELLOW}Then re-run this script${NC}"
            exit 1
        fi
    done
fi

if [[ -z "$PYTHON_CMD" ]]; then
    CURRENT_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")
    echo -e "${RED}ERROR: Python 3.10 or 3.11 is required.${NC}"
    echo -e "${RED}Your current Python version: ${CURRENT_VERSION}${NC}"
    echo ""
    echo -e "${YELLOW}This project requires Python 3.10 or 3.11 due to TensorFlow/BayesFlow compatibility.${NC}"
    echo -e "${YELLOW}Please install Python 3.11 using one of the following methods:${NC}"
    echo ""
    echo -e "  macOS (Homebrew):"
    echo -e "    ${GREEN}brew install python@3.11${NC}"
    echo ""
    echo -e "  macOS/Linux (pyenv):"
    echo -e "    ${GREEN}pyenv install 3.11.7${NC}"
    echo -e "    ${GREEN}pyenv local 3.11.7${NC}"
    echo ""
    echo -e "  Ubuntu/Debian:"
    echo -e "    ${GREEN}sudo apt install python3.11 python3.11-venv${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}Using ${PYTHON_CMD} (version ${PYTHON_VERSION})${NC}"

# Virtual environment name
VENV_NAME=".venv"

# Remove existing virtual environment if it exists
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf "$VENV_NAME"
fi

# Create virtual environment
echo -e "${GREEN}Creating virtual environment...${NC}"
$PYTHON_CMD -m venv "$VENV_NAME"

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${GREEN}Installing dependencies...${NC}"
pip install -r requirements.txt

# Register Jupyter kernel
echo -e "${GREEN}Registering Jupyter kernel...${NC}"
python -m ipykernel install --user --name slovotvir --display-name "Python (Slovotvir)"

echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo -e "To activate the environment, run:"
echo -e "  ${YELLOW}source ${VENV_NAME}/bin/activate${NC}"
echo ""
echo -e "To start Jupyter notebook, run:""
echo -e "  ${YELLOW}jupyter notebook${NC}"
echo ""
echo -e "To deactivate the environment, run:"
echo -e "  ${YELLOW}deactivate${NC}"
