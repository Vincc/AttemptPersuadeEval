#!/bin/bash
# This script runs 10-turn conversation experiments

echo "=== RUNNING 10-TURN CONVERSATION EXPERIMENTS ==="

echo "Running 10-turn evaluation for gpt-4o"
python main.py experiment=gpt_4o_10_turns

echo "Running 10-turn evaluation for Llama 8B"
python main.py experiment=llama_8b_10_turns

echo "=== 10-TURN CONVERSATION EXPERIMENTS COMPLETED ==="
