"""Tests for cleanup module."""

from unittest.mock import MagicMock, patch

from dictation.cleanup.prompts import (
    build_cleanup_message,
    build_cleanup_system,
    build_few_shot_messages,
)
from dictation.cleanup.ollama import OllamaCleanup


class TestPrompts:
    def test_system_prompt_contains_key_rules(self):
        system = build_cleanup_system()
        assert "NEVER" in system
        assert "Answer questions" in system
        assert "capitalization" in system.lower()

    def test_few_shot_messages_alternating_roles(self):
        messages = build_few_shot_messages()
        assert len(messages) >= 4  # at least 2 examples
        for i, msg in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role

    def test_few_shot_question_not_answered(self):
        """Verify the few-shot examples include a question that's formatted, not answered."""
        messages = build_few_shot_messages()
        for i in range(0, len(messages), 2):
            user = messages[i]["content"]
            assistant = messages[i + 1]["content"]
            if "meeting" in user and "time" in user:
                assert "?" in assistant
                assert "start" in assistant

    def test_cleanup_message_without_context(self):
        msg = build_cleanup_message("hello world")
        assert msg == "hello world"

    def test_cleanup_message_returns_transcript_verbatim(self):
        msg = build_cleanup_message("and then I left")
        assert msg == "and then I left"

    def test_system_prompt_includes_context_hints(self):
        system = build_cleanup_system(
            {
                "app_name": "Slack",
                "field_text": "following up on the launch timing",
                "recent_utterances": ["let's keep it short", "mention the beta users"],
                "dictionary_hint": "Custom vocabulary: Ollama, Whisper",
                "has_backtrack": True,
            }
        )
        assert "Slack" in system
        assert "launch timing" in system
        assert "beta users" in system
        assert "Ollama" in system
        assert "corrected themselves" in system

    def test_system_prompt_uses_code_app_instruction(self):
        system = build_cleanup_system({"app_name": "Cursor"})
        assert "code/editor style" in system.lower()
        assert "technical casing" in system.lower()


class TestOllamaCleanup:
    def test_empty_input_passthrough(self):
        cleanup = OllamaCleanup()
        assert cleanup.cleanup("") == ""
        assert cleanup.cleanup("   ") == "   "

    @patch("dictation.cleanup.ollama.httpx.Client")
    def test_successful_cleanup(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Hello, world."}
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        cleanup = OllamaCleanup()
        cleanup._client = mock_client
        result = cleanup.cleanup("hello world")
        assert result == "Hello, world."

    @patch("dictation.cleanup.ollama.httpx.Client")
    def test_uses_chat_api_with_keep_alive(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Test."}
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        cleanup = OllamaCleanup()
        cleanup._client = mock_client
        cleanup.cleanup("test")

        call_args = mock_client.post.call_args
        assert "/api/chat" in call_args[0][0]
        payload = call_args[1]["json"]
        assert "messages" in payload
        assert payload["keep_alive"] == "10m"

    @patch("dictation.cleanup.ollama.httpx.Client")
    def test_any_exception_returns_raw(self, mock_client_cls):
        """Fail-open catches ANY exception, not just specific ones."""
        mock_client = MagicMock()
        mock_client.post.side_effect = ValueError("bad json")
        mock_client_cls.return_value = mock_client

        cleanup = OllamaCleanup()
        cleanup._client = mock_client
        result = cleanup.cleanup("hello world")
        assert result == "hello world"

    @patch("dictation.cleanup.ollama.httpx.Client")
    def test_no_context_in_message(self, mock_client_cls):
        """Verify cleanup sends only the transcript, no context prefix."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "And then I left."}
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        cleanup = OllamaCleanup()
        cleanup._client = mock_client
        cleanup.cleanup("and then i left")

        call_args = mock_client.post.call_args
        messages = call_args[1]["json"]["messages"]
        last_user_msg = messages[-1]["content"]
        assert last_user_msg == "and then i left"
        assert "Context:" not in last_user_msg

    @patch("dictation.cleanup.ollama.httpx.Client")
    def test_context_is_embedded_in_system_prompt(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "I'll send the update soon."}
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        cleanup = OllamaCleanup()
        cleanup._client = mock_client
        cleanup.cleanup(
            "ill send the update soon",
            context={
                "app_name": "Mail",
                "field_text": "thanks again for the quick turnaround",
                "recent_utterances": ["we should sound polished here"],
                "dictionary_hint": "Custom vocabulary: Ollama",
            },
        )

        call_args = mock_client.post.call_args
        system_msg = call_args[1]["json"]["messages"][0]["content"]
        assert "Mail" in system_msg
        assert "turnaround" in system_msg
        assert "polished" in system_msg
        assert "Ollama" in system_msg

    @patch("dictation.cleanup.ollama.httpx.Client")
    def test_divergent_output_falls_back_to_raw(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Please review the attached project plan and share feedback tomorrow."}
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        cleanup = OllamaCleanup()
        cleanup._client = mock_client
        result = cleanup.cleanup("hello world this is a test")
        assert result == "hello world this is a test"

    @patch("dictation.cleanup.ollama.httpx.Client")
    def test_warmup_returns_bool(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        cleanup = OllamaCleanup()
        cleanup._client = mock_client
        assert cleanup.warmup() is True

    @patch("dictation.cleanup.ollama.httpx.Client")
    def test_warmup_returns_false_on_failure(self, mock_client_cls):
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        cleanup = OllamaCleanup()
        cleanup._client = mock_client
        assert cleanup.warmup() is False
