#!/usr/bin/env bash

# Run the linter
pylint model.py estimator.py example.py

# Run mypy
mypy --ignore-missing-imports --strict --no-warn-unused-ignores model.py estimator.py example.py
