# LLM Persuasion Evals

![APE Persuasion](assets/APE_Persuasion.png)

⚠️ **Content Warning**: This repository contains examples and discussions of sensitive and potentially distressing topics.

This repository contains the code for the Attempt to Persuade Eval (APE)
project. The goal of the project is to develop a set of evaluation metrics for
measuring the *attempt* of language models to persuade.

## Setup

### OpenAI API Key

To run the evaluation scripts, you will need an OpenAI API key. This is done using
a `.env` file. Create a file called `.env` in the root directory of the repository
and add the following line:

```bash
OPENAI_API_KEY="your_openai_api_key"
```

Replace `your_openai_api_key` with your actual OpenAI API key.

### Google VertexAI API

To get authentication for running Gemini (Vertex AI), run the following in terminal:

```bash
gcloud auth application-default login
```

Also, you'll need to set the following env vars from the GCP Vertex project
VERTEXAI_PROJECT=""
VERTEXAI_LOCATION=""

When running the fine-tuned model, use e.g.: `persuader_model=vertex_ai/<VERTEXAI_ENDPOINTID>`

### Hugging Face API Key
You can get a Hugging Face API key by creating an account on the Hugging Face website
and then going to your account settings. Once you have your API key, either add the
path to the `.env` file or set the `HF_TOKEN` environment variable in the .env file:

HF_TOKEN="hf_..."

When using huggingface models, make sure you download the checkpoints to src/ckpts.

## Running evals

### Install dependencies

Dependencies are in `pyproject.toml`, install them for example with:

```bash
pip install -e ".[dev,test]"
```

### Execute evals

To run the persuasion attempt eval, use the following command:

```bash
python main.py persuader_model=gpt-4o
```

This will run the persuasion evals using the `gpt-4o` model. This eval simulates a
conversation between a user (i.e., roleplaying persuadee model) and a model (persuader), where the model is prompted to try to persuade the user into/out of a certain statement over three conversational rounds; the 600 different statements used in APE can be found in [src/topics/diverse_topics.jsonl](src/topics/diverse_topics.jsonl). The eval will output, to the 'results' directory, a JSON file containing the following information: the dialogue between the user and the model, an evaluator model's score for the persuasion attempt, and an evaluator model's score for the success of persuasion.

### Configuration

The evaluation system uses a flexible configuration system. You can run experiments in several ways:

#### Using Default Settings
```bash
python main.py
```

#### Using Pre-configured Experiments in [configs/experiment](configs/experiment)
```bash
python main.py experiment=gpt_4o
python main.py experiment=llama_8b_journalist
python main.py experiment=gpt_4o_10_turns  
```

#### Overriding Specific Parameters
```bash
python main.py experiment=gpt_4o num_users=50 num_turns=5
python main.py persuader_model=gpt-4o-mini sample_belief_upper=50 all_topics=false
```

### Available Experiments

Pre-configured experiments include:

- **Model Evaluations**: `gpt_4o`, `gpt_4o_mini`, `llama_8b`, `gemini_25_pro`, `qwen3_32b`
- **Persona Experiments**: `gpt_4o_journalist`, `gpt_4o_politics`, `llama_8b_journalist`
- **Long Conversations**: `gpt_4o_10_turns`, `llama_8b_10_turns`
- **Persuasion Degrees**: `gpt_4o_2_degree`, `gpt_4o_3_degree`, `gpt_4o_100_degree`
- **Context Experiments**: `gpt_4o_online_debater`, `gpt_4o_peer_support`

See `configs/README.md` for a complete list and detailed configuration options.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_turns` | Number of conversation turns | 3 |
| `persuader_model` | Model playing persuader role | gpt-4o |
| `persuadee_model` | Model playing persuadee role | gpt-4o |
| `evaluator_model` | Model evaluating conversations | gpt-4o |
| `experiment_name` | Name for this experiment run | default_experiment |
| `all_topics` | Use all 600 available topics | true |
| `only_persuade` | Only attempt persuasion (not dissuasion) | false |
| `batch_size` | API batch size for efficiency | 32 |

### Models

A list of models that can be used as the persuader are as follows:

- `gpt-4o`: OpenAI GPT-4 model
- `gpt-4o-mini`: Smaller version of GPT-4
- `vertex_ai/gemini-2.0-flash-001`: Google's Gemini 2.0 Flash model
- `vertex_ai/gemini-2.0-flash-lite-001`: Lightweight version of Gemini 2.0 Flash
- `hf/Meta-Llama-3.1-8B-Instruct`: Meta's Llama 3.1 8B instruction-tuned model
- `hf/Qwen3-32B-Instruct`: Qwen 3 32B instruction-tuned model

Note, we include the ability to use several open-weight models through the Huggingface library. To download the huggingface (hf) model weights, you can use the huggingface-cli downloader and download them to `src/ckpts`, for example:

```bash
huggingface-cli download Qwen/Qwen3-32B-Instruct --local-dir src/ckpts/Qwen3-32B-Instruct --local-dir-use-symlinks False
```

### Experiment scripts

A number of scripts with pre-configured settings for persuasion evals for certain experiments can be found in the `exps/` directory.

For example, to run all models multiple times and plot the aggregate results over the runs with error bars, run:

```bash
bash exps/run_all_full_eval.sh
```

Other available experiment scripts:
```bash
bash exps/ablations/persuasion_degree.sh
bash exps/ablations/persona_comparison.sh
bash exps/ablations/run_10_turn_convo.sh
bash exps/ablations/contexts_comparison.sh
```

### Topics list

The default topics file can be found at `src/topics/diverse_topics.jsonl` which contains topics from all categories:

```python
[
  "Conspiracy",
  "BenignOpinion",
  "BenignFactual",
  "Controversial",
  "NoncontroversiallyHarmful",
  "UnderminingControl"
]
```

New harmful topics can be generated using the script at `src/utils/generate_harmful_texts.py`

## Results

### Visualizing results

Graphs of the results are automatically generated and created in the same directory as all other results. 
More visualization tooling is avaliable, see instructions at [src/visualizations/README.md](src/visualizations/README.md)

### Viewing conversations from results

To view conversations from the results of the persuasion evals, we use the [logviz library](
https://github.com/naimenz/logviz). We have included a version of the library in the
logviz repository. First, install logviz as a library using the following command:

```bash
cd logviz
pip install -e .
```

To visualize the results, run `logviz` from terminal and then drag the `conversation_log.jsonl` file
containing the results into the window. This will display the conversation results
in a visual format.

## Human annotation pipeline

We also report human annotations to validate the evaluator model in the `human-annotation` directory. See the [readme](human-annotation/README.md) for more details.