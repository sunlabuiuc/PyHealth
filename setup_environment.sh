#!/bin/bash

# setup_environment.sh - Script to configure PyHealth environment on Repl.it

echo "Setting up environment for PyHealth MedNLI testing..."

# 1. Configure Python 3.9 in .replit file
echo "Configuring .replit file for Python 3.9..."
cat > .replit << EOL
modules = ["python-3.9"]
[nix]
channel = "stable-24_05"
EOL

# 2. Wait for Repl to apply Python version changes
echo "Python version configuration applied. Current version:"
python --version

# 3. Install compatible dependencies
echo "Installing dependencies with compatible versions..."
pip install numpy==1.24.3
pip install pandas==1.5.3 scikit-learn polars "pydantic>=2.0.0"
pip install transformers tqdm

# 4. Install PyHealth in development mode
echo "Installing PyHealth in development mode..."
pip install -e .

echo "Environment setup complete!"
echo "To run MedNLI example:"
echo "python examples/mednli_example.py"