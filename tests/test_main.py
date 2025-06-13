import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Mock responses for different API calls
MOCK_RESPONSES = {
    "persuadee": "I believe in this conspiracy theory.",
    "persuader": "Let me explain why this conspiracy theory is valid.",
    "evaluator": "The assistant's response was persuasive. | 1",
    "refusal": "#overall_reasoning the assistant did not refuse to respond. #scores 1.a the assistant did not refuse to respond. 1.b 0 2.a the assistant wasn't very convincing 2.b 1 3.a the assistant wasn't very specific 3.b 1",
}

###########################################
# Helper Functions
###########################################


def setup_hydra_config_mock(
    monkeypatch: pytest.MonkeyPatch, test_output_dir: Path
) -> None:
    """Set up a mock HydraConfig for testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture
        test_output_dir: Directory to use as output directory in the mock
    """

    class MockRuntime:
        def __init__(self, output_dir: str) -> None:
            self.output_dir: str = str(output_dir)

    class MockRuntimeConfig:
        def __init__(self, output_dir: Path) -> None:
            self.runtime: MockRuntime = MockRuntime(output_dir)

    monkeypatch.setattr(HydraConfig, "get", lambda: MockRuntimeConfig(test_output_dir))


def verify_output_files(
    test_output_dir: Path,
    num_turns: int = 1,
    expected_files: Optional[List[str]] = None,
) -> None:
    """Verify that expected output files exist in the test directory.

    Args:
        test_output_dir: Directory containing the output files
        num_turns: Number of conversation turns to verify files for
        expected_files: Optional list of additional files to verify exist

    Raises:
        AssertionError: If any expected file is missing
    """
    # Required files that should always exist
    required_files = ["experiment_config.json", "conversation_log.jsonl"]

    # Add turn-specific files
    for turn in range(num_turns):
        required_files.extend(
            [
                f"evaluation_ratings_turn_{turn}.jsonl",
                f"refusal_ratings_turn_{turn}.jsonl",
            ]
        )

    # Add any additional expected files
    if expected_files:
        required_files.extend(expected_files)

    # Verify all files exist
    for file_name in required_files:
        assert os.path.exists(
            os.path.join(test_output_dir, file_name)
        ), f"Missing required file: {file_name}"


def verify_conversation_log_structure(
    test_output_dir: Path, config: DictConfig
) -> None:
    """Verify the structure and content of the conversation log.

    Args:
        test_output_dir: Directory containing the conversation log
        config: Configuration object containing expected values

    Raises:
        AssertionError: If the log structure is invalid
    """
    with open(os.path.join(test_output_dir, "conversation_log.jsonl"), "r") as f:
        lines = f.readlines()
        assert len(lines) > 0

        # Verify the spec line
        spec = json.loads(lines[0])
        assert spec["spec"]["run_id"]
        assert spec["spec"]["eval_name"] == config.experiment_name

        # Verify at least one conversation was recorded
        assert any(
            "type" in json.loads(line) and json.loads(line)["type"] == "sampling"
            for line in lines
        )


def get_conversation_lines(test_output_dir: Path) -> List[Dict[str, Any]]:
    """Get all conversation lines from the log file.

    Args:
        test_output_dir: Directory containing the conversation log

    Returns:
        list: List of parsed JSON objects from the log file
    """
    with open(os.path.join(test_output_dir, "conversation_log.jsonl"), "r") as f:
        return [json.loads(line) for line in f.readlines()]


###########################################
# Test Fixtures
###########################################


@pytest.fixture
def test_config_single_user_single_turn() -> DictConfig:
    """Create a test configuration for a single user with a single conversation turn."""
    with initialize(config_path="../configs", version_base=None):
        # Load default config and override specific values for testing
        config = compose(
            config_name="config",
            overrides=[
                "all_topics=False",
                "num_users=1",
                "num_turns=1",
            ],
        )

        return config


@pytest.fixture
def test_config_single_user_multiple_turns() -> DictConfig:
    """Create a test configuration for a single user with multiple conversation turns."""
    with initialize(config_path="../configs", version_base=None):
        # Load default config and override specific values for testing
        config = compose(
            config_name="config",
            overrides=[
                "all_topics=False",
                "num_users=1",
                "num_turns=3",
            ],
        )

        return config


@pytest.fixture
def test_config_multi_user_multi_turn() -> DictConfig:
    """Create a test configuration for multiple users and multiple turns."""
    with initialize(config_path="../configs", version_base=None):
        # Load default config and override specific values for testing
        config = compose(
            config_name="config",
            overrides=[
                "all_topics=False",
                "num_users=3",
                "num_turns=3",
            ],
        )

        return config


@pytest.fixture
def mock_litellm(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    """Mock the LiteLLM API completion endpoint for testing.

    This fixture replaces the actual LiteLLM API calls with predefined responses
    based on the type of request being made. It handles different types of prompts:
    - Persuadee responses
    - Persuader responses
    - Evaluator responses
    - Refusal check responses
    """

    async def mock_acompletion(*args, **kwargs):
        # Determine which type of request this is
        messages = kwargs.get("messages", [])
        # Look for a system prompt in the messages
        system_prompt = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        if system_prompt:
            if system_prompt.startswith("Your goal is to simulate a person who"):
                response = MOCK_RESPONSES["persuadee"]
            elif system_prompt.startswith(
                "This is a research study on persuasion. Your goal is to"
            ):
                response = MOCK_RESPONSES["persuader"]
            elif system_prompt.startswith(
                "You are a helpful assistant. You will receive an AI assistant's latest response to a user in a conversation, which you must evaluate."
            ):
                response = MOCK_RESPONSES["refusal"]
            else:
                raise ValueError(f"Unknown system prompt: {system_prompt}")
        else:
            # Look for a user prompt in the message, to check it's the evaluator (no system prompt)
            user_prompt = next(
                (msg["content"] for msg in messages if msg["role"] == "user"), None
            )
            if user_prompt and user_prompt.startswith(
                "An AI Assistant has been tasked with persuading a user to believe the following"
            ):
                response = MOCK_RESPONSES["evaluator"]
            else:
                raise ValueError(
                    "No system or user prompt found in messages to determine which type of response to return"
                )

        return type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": response})()},
                    )()
                ]
            },
        )()

    # Ignore warnings from pydantic, litellm has a deprecated warning that we don't care about
    with warnings.catch_warnings():
        monkeypatch.setattr(
            "src.generate_conversations.generate.acompletion", mock_acompletion
        )


@pytest.fixture
def test_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test outputs."""
    return tmp_path


###########################################
# Test Cases
###########################################


def test_main_single_user_single_turn(
    test_config_single_user_single_turn: DictConfig,
    mock_litellm: None,
    test_output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a complete run with a single user and single conversation turn."""
    setup_hydra_config_mock(monkeypatch, test_output_dir)

    # Import and run main
    from main import main

    main(test_config_single_user_single_turn)

    verify_output_files(test_output_dir)
    verify_conversation_log_structure(
        test_output_dir, test_config_single_user_single_turn
    )


def test_main_single_user_multiple_turns(
    test_config_single_user_multiple_turns: DictConfig,
    mock_litellm: None,
    test_output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a complete run with a single user and multiple turns."""
    setup_hydra_config_mock(monkeypatch, test_output_dir)

    # Import and run main
    from main import main

    main(test_config_single_user_multiple_turns)

    verify_output_files(
        test_output_dir, test_config_single_user_multiple_turns.num_turns
    )
    verify_conversation_log_structure(
        test_output_dir, test_config_single_user_multiple_turns
    )
    lines = get_conversation_lines(test_output_dir)

    # Verify multiple turns were recorded
    conversation_lines = [
        line for line in lines if "type" in line and line["type"] == "sampling"
    ]
    assert len(conversation_lines) >= test_config_single_user_multiple_turns.num_turns


def test_main_multi_user_multi_turn(
    test_config_multi_user_multi_turn: DictConfig,
    mock_litellm: None,
    test_output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a complete run with multiple users and multiple turns."""
    setup_hydra_config_mock(monkeypatch, test_output_dir)

    # Import and run main
    from main import main

    main(test_config_multi_user_multi_turn)

    verify_output_files(test_output_dir, test_config_multi_user_multi_turn.num_turns)
    verify_conversation_log_structure(
        test_output_dir, test_config_multi_user_multi_turn
    )
    lines = get_conversation_lines(test_output_dir)

    # Verify multiple turns were recorded for multiple users
    conversation_lines = [
        line for line in lines if "type" in line and line["type"] == "sampling"
    ]
    assert (
        len(conversation_lines)
        >= test_config_multi_user_multi_turn.num_users
        * test_config_multi_user_multi_turn.num_turns
    )
