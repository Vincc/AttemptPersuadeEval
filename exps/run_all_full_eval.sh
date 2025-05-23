#!/bin/bash
# This script runs all main model evaluations 

echo "=== RUNNING ALL FULL EVALUATIONS ==="

echo "Running full evaluation for gpt-4o"
python main.py experiment=gpt_4o

echo "Running full evaluation for gpt-4o-jb"
python main.py experiment=gpt_4o_jb

echo "Running full evaluation for vertex_ai/gemini-2.5-pro-preview-03-25"
python main.py experiment=gemini_25_pro

echo "Running full evaluation for gpt-4o-mini"
python main.py experiment=gpt_4o_mini

echo "Running full evaluation for vertex_ai/gemini-2.0-flash-001"
python main.py experiment=gemini_flash_001

echo "Running full evaluation for Llama 8B"
python main.py experiment=llama_8b

echo "Running full evaluation for hf/Qwen3-32B"
python main.py experiment=qwen3_32b

echo "=== ALL EVALUATIONS COMPLETED ===" 

echo "=== Running visualization scripts to generate plots ==="
bash src/visualizations/generate_filtered_plots.sh --results_dir results/aggregate_plots --output_dir results/aggregate_plots/filtered_model_plots