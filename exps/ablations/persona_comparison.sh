#!/bin/bash
# This script runs persona comparison experiments

echo "=== RUNNING PERSONA COMPARISON EXPERIMENTS ==="

echo "Running JOURNALIST persona evaluation for GPT-4o"
python main.py experiment=gpt_4o_journalist

echo "Running POLITICAL persona evaluation for GPT-4o"
python main.py experiment=gpt_4o_politics

echo "Running JOURNALIST persona evaluation for Llama 8B"
python main.py experiment=llama_8b_journalist

echo "Running POLITICAL persona evaluation for Llama 8B"
python main.py experiment=llama_8b_politics

echo "=== PERSONA COMPARISON EXPERIMENTS COMPLETED ===" 