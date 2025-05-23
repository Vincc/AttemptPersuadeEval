# Configuration System

This directory contains configuration files for the persuasion evaluation experiments.

## Structure

- `config.yaml` - Default configuration with all parameters
- `experiment/` - Experiment-specific configurations that override defaults

## Quick Start

### Running with Default Settings

```bash
python main.py
```

### Running Specific Experiments

```bash
python main.py experiment=gpt_4o
python main.py experiment=llama_8b_journalist
```

### Overriding Parameters

```bash
python main.py experiment=gpt_4o num_users=50 num_turns=5
python main.py persuader_model=gpt-4o-mini sample_belief_upper=50
```

## Available Experiments

#### Main Model Evaluations
- `gpt_4o` - GPT-4o evaluation
- `gpt_4o_jb` - GPT-4o jailbroken model
- `gpt_4o_mini` - GPT-4o Mini
- `gemini_25_pro` - Gemini 2.5 Pro
- `gemini_flash_001` - Gemini Flash 001
- `llama_8b` - Llama 8B
- `qwen3_32b` - Qwen3 32B

#### Persona Experiments
- `gpt_4o_journalist` - GPT-4o with journalist persona
- `gpt_4o_politics` - GPT-4o with political persona
- `llama_8b_journalist` - Llama 8B with journalist persona
- `llama_8b_politics` - Llama 8B with political persona

#### Long Conversation Experiments
- `gpt_4o_10_turns` - GPT-4o with 10-turn conversations
- `llama_8b_10_turns` - Llama 8B with 10-turn conversations

#### Persuasion Degree Ablations
- `gpt_4o_2_degree` - GPT-4o with 2-degree persuasion scale
- `gpt_4o_3_degree` - GPT-4o with 3-degree persuasion scale
- `gpt_4o_100_degree` - GPT-4o with 100-degree persuasion scale
- `llama_8b_2_degree` - Llama 8B with 2-degree persuasion scale
- `llama_8b_3_degree` - Llama 8B with 3-degree persuasion scale
- `llama_8b_100_degree` - Llama 8B with 100-degree persuasion scale

#### Situational Context Experiments
- `gpt_4o_online_debater` - GPT-4o with online debater context
- `gpt_4o_peer_support` - GPT-4o with peer support context
- `llama_8b_online_debater` - Llama 8B with online debater context

## Experiment Scripts

Run multiple experiments with these convenience scripts:

```bash
# Main model evaluations (3 runs each)
bash exps/run_all_full_eval.sh

# Persuasion degree ablations
bash exps/ablations/persuasion_degree.sh

# Persona comparisons
bash exps/ablations/persona_comparison.sh

# 10-turn conversations
bash exps/ablations/run_10_turn_convo.sh

# Context comparisons
bash exps/ablations/contexts_comparison.sh
```

## Creating New Experiments

1. Create a new YAML file in `configs/experiment/`
2. Start with `# @package _global_`
3. Override only the parameters that differ from defaults

Example:
```yaml
# @package _global_
# My Custom Experiment

persuader_model: gpt-4o-mini
num_turns: 5
experiment_name: my_custom_experiment
results_dir: results/my_experiment
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_users` | Number of simulated users | 100 |
| `num_turns` | Conversation turns per user | 3 |
| `persuader_model` | Model playing persuader role | gpt-4o |
| `persuadee_model` | Model playing persuadee role | gpt-4o |
| `evaluator_model` | Model evaluating conversations | gpt-4o |
| `experiment_name` | Name for this experiment run | default_experiment |
| `all_topics` | Use all available topics | false |
| `only_persuade` | Only attempt persuasion (not dissuasion) | false |
| `batch_size` | API batch size for efficiency | 32 | 