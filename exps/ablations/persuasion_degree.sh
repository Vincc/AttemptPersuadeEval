#!/bin/bash
# This script runs persuasion degree ablation experiments

echo "=== RUNNING PERSUASION DEGREE EXPERIMENTS ==="

echo "Running 2 degree evaluation for GPT-4o"
python main.py experiment=gpt_4o_2_degree

echo "Running 3 degree evaluation for GPT-4o"
python main.py experiment=gpt_4o_3_degree

echo "Running 100 degree evaluation for GPT-4o"
python main.py experiment=gpt_4o_100_degree

echo "Running 2 degree evaluation for Llama 8B"
python main.py experiment=llama_8b_2_degree

echo "Running 3 degree evaluation for Llama 8B"
python main.py experiment=llama_8b_3_degree

echo "Running 100 degree evaluation for Llama 8B"
python main.py experiment=llama_8b_100_degree

echo "=== PERSUASION DEGREE EXPERIMENTS COMPLETED ==="

echo "=== Running visualization scripts to generate plots ==="
bash src/visualizations/generate_persuasion_degree_plots.sh --results_dir results/persuasion_degree --output_dir results/persuasion_degree/plots
