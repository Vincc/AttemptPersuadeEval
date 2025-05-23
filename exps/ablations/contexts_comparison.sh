#!/bin/bash
# This script runs situational context experiments

echo "=== RUNNING CONTEXTS COMPARISON EXPERIMENTS ==="

echo "Running Online Debater context for GPT-4o"
python main.py experiment=gpt_4o_online_debater

echo "Running Peer Support context for GPT-4o"
python main.py experiment=gpt_4o_peer_support

echo "Running Virtual Influencer context for GPT-4o"
python main.py experiment=gpt_4o_virtual_influencer

echo "Running Political Strategist context for GPT-4o"
python main.py experiment=gpt_4o_political_strategist

echo "Running Controversial Topic Guide context for GPT-4o"
python main.py experiment=gpt_4o_controversial_topic_guide

echo "Running Online Debater context for Llama 8B"
python main.py experiment=llama_8b_online_debater

echo "Running Virtual Influencer context for Llama 8B"
python main.py experiment=llama_8b_virtual_influencer

echo "Running Peer Support context for Llama 8B"
python main.py experiment=llama_8b_peer_support

echo "Running Political Strategist context for Llama 8B"
python main.py experiment=llama_8b_political_strategist

echo "Running Controversial Topic Guide context for Llama 8B"
python main.py experiment=llama_8b_controversial_topic_guide

echo "=== CONTEXTS COMPARISON EXPERIMENTS COMPLETED ==="

echo "=== Running visualization scripts to generate plots ==="
bash src/visualizations/generate_situational_plots.sh
