#!/bin/bash

DELETE_ENV=true
DELETE_LOGS=false

{
  /opt/homebrew/bin/python3.12 -m venv venv
  source venv/bin/activate

  pip install --upgrade pip
  pip install -e .
  pip install pytest x-transformers

  rm -rf /tmp/physionet2012
  rm -rf ~/.cache/pyhealth/
  rm -rf ~/Library/Caches/pyhealth/
  rm -rf output/
} > install.log 2>&1

{
  pytest tests/core/test_physionet2012.py \
         tests/core/test_physionet_mortality.py \
         tests/core/test_duett.py \
         tests/core/test_duett_processor.py -v
} > test.log 2>&1

{
  python examples/physionet2012_mortality_duett.py

  rm -rf /tmp/physionet2012
  rm -rf output/
} > demo.log 2>&1

if [ "$DELETE_ENV" = true ]; then
  rm -rf venv
fi

if [ "$DELETE_LOGS" = true ]; then
  rm -f install.log test.log demo.log
fi