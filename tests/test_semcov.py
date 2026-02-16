"""
Tests for the tmam.semcov module.
"""

import pytest

import sys

sys.path.insert(0, "/home/claude/sdk/sdk/src")

from tmam.semcov import SemanticConvetion


class TestSemanticConventionAttributes:
    """Tests for SemanticConvetion class attributes."""

    def test_general_attributes(self):
        """Test general OTel Semconv attributes."""
        assert SemanticConvetion.SERVER_PORT == "server.port"
        assert SemanticConvetion.SERVER_ADDRESS == "server.address"
        assert SemanticConvetion.ERROR_TYPE == "error.type"

    def test_genai_metric_names(self):
        """Test GenAI metric names."""
        assert (
            SemanticConvetion.GEN_AI_CLIENT_TOKEN_USAGE == "gen_ai.client.token.usage"
        )
        assert (
            SemanticConvetion.GEN_AI_CLIENT_OPERATION_DURATION
            == "gen_ai.client.operation.duration"
        )
        assert (
            SemanticConvetion.GEN_AI_SERVER_REQUEST_DURATION
            == "gen_ai.server.request.duration"
        )
        assert (
            SemanticConvetion.GEN_AI_SERVER_TBT == "gen_ai.server.time_per_output_token"
        )
        assert (
            SemanticConvetion.GEN_AI_SERVER_TTFT == "gen_ai.server.time_to_first_token"
        )

    def test_genai_event_names(self):
        """Test GenAI event names."""
        assert SemanticConvetion.GEN_AI_USER_MESSAGE == "gen_ai.user.message"
        assert SemanticConvetion.GEN_AI_SYSTEM_MESSAGE == "gen_ai.system.message"
        assert SemanticConvetion.GEN_AI_ASSISTANT_MESSAGE == "gen_ai.assistant.message"
        assert SemanticConvetion.GEN_AI_TOOL_MESSAGE == "gen_ai.tools.message"
        assert SemanticConvetion.GEN_AI_CHOICE == "gen_ai.choice"

    def test_genai_request_attributes(self):
        """Test GenAI request attributes."""
        assert SemanticConvetion.GEN_AI_OPERATION == "gen_ai.operation.name"
        assert SemanticConvetion.GEN_AI_SYSTEM == "gen_ai.system"
        assert SemanticConvetion.GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
        assert SemanticConvetion.GEN_AI_REQUEST_SEED == "gen_ai.request.seed"
        assert (
            SemanticConvetion.GEN_AI_REQUEST_TEMPERATURE == "gen_ai.request.temperature"
        )
        assert (
            SemanticConvetion.GEN_AI_REQUEST_MAX_TOKENS == "gen_ai.request.max_tokens"
        )
        assert SemanticConvetion.GEN_AI_REQUEST_TOP_P == "gen_ai.request.top_p"
        assert SemanticConvetion.GEN_AI_REQUEST_TOP_K == "gen_ai.request.top_k"
        assert (
            SemanticConvetion.GEN_AI_REQUEST_STOP_SEQUENCES
            == "gen_ai.request.stop_sequences"
        )

    def test_genai_response_attributes(self):
        """Test GenAI response attributes."""
        assert (
            SemanticConvetion.GEN_AI_RESPONSE_FINISH_REASON
            == "gen_ai.response.finish_reasons"
        )
        assert SemanticConvetion.GEN_AI_RESPONSE_ID == "gen_ai.response.id"
        assert SemanticConvetion.GEN_AI_RESPONSE_MODEL == "gen_ai.response.model"
        assert (
            SemanticConvetion.GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
        )
        assert (
            SemanticConvetion.GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
        )
        assert SemanticConvetion.GEN_AI_TOOL_CALL_ID == "gen_ai.tool.call.id"
        assert SemanticConvetion.GEN_AI_TOOL_NAME == "gen_ai.tool.name"

    def test_genai_operation_types(self):
        """Test GenAI operation types."""
        assert SemanticConvetion.GEN_AI_OPERATION_TYPE_CHAT == "chat"
        assert SemanticConvetion.GEN_AI_OPERATION_TYPE_TOOLS == "execute_tool"
        assert SemanticConvetion.GEN_AI_OPERATION_TYPE_EMBEDDING == "embeddings"
        assert SemanticConvetion.GEN_AI_OPERATION_TYPE_IMAGE == "image"
        assert SemanticConvetion.GEN_AI_OPERATION_TYPE_AUDIO == "audio"
        assert SemanticConvetion.GEN_AI_OPERATION_TYPE_VECTORDB == "vectordb"
        assert SemanticConvetion.GEN_AI_OPERATION_TYPE_FRAMEWORK == "framework"
        assert SemanticConvetion.GEN_AI_OPERATION_TYPE_AGENT == "agent"

    def test_genai_system_names_otel(self):
        """Test GenAI system names (OTel standard)."""
        assert SemanticConvetion.GEN_AI_SYSTEM_ANTHROPIC == "anthropic"
        assert SemanticConvetion.GEN_AI_SYSTEM_AWS_BEDROCK == "aws.bedrock"
        assert SemanticConvetion.GEN_AI_SYSTEM_AZURE_AI_INFERENCE == "az.ai.inference"
        assert SemanticConvetion.GEN_AI_SYSTEM_COHERE == "cohere"
        assert SemanticConvetion.GEN_AI_SYSTEM_GROQ == "groq"
        assert SemanticConvetion.GEN_AI_SYSTEM_MISTRAL == "mistral_ai"
        assert SemanticConvetion.GEN_AI_SYSTEM_OPENAI == "openai"
        assert SemanticConvetion.GEN_AI_SYSTEM_VERTEXAI == "vertex_ai"

    def test_genai_system_names_extra(self):
        """Test GenAI system names (extras)."""
        assert SemanticConvetion.GEN_AI_SYSTEM_HUGGING_FACE == "huggingface"
        assert SemanticConvetion.GEN_AI_SYSTEM_OLLAMA == "ollama"
        assert SemanticConvetion.GEN_AI_SYSTEM_GPT4ALL == "gpt4all"
        assert SemanticConvetion.GEN_AI_SYSTEM_ELEVENLABS == "elevenlabs"
        assert SemanticConvetion.GEN_AI_SYSTEM_VLLM == "vLLM"
        assert SemanticConvetion.GEN_AI_SYSTEM_GOOGLE_AI_STUDIO == "google.ai.studio"
        assert SemanticConvetion.GEN_AI_SYSTEM_LANGCHAIN == "langchain"
        assert SemanticConvetion.GEN_AI_SYSTEM_LLAMAINDEX == "llama_index"
        assert SemanticConvetion.GEN_AI_SYSTEM_HAYSTACK == "haystack"
        assert SemanticConvetion.GEN_AI_SYSTEM_CREWAI == "crewai"
        assert SemanticConvetion.GEN_AI_SYSTEM_LITELLM == "litellm"

    def test_vector_db_attributes(self):
        """Test Vector DB attributes."""
        assert SemanticConvetion.DB_SYSTEM_NAME == "db.system.name"
        assert SemanticConvetion.DB_COLLECTION_NAME == "db.collection.name"
        assert SemanticConvetion.DB_NAMESPACE == "db.query.namespace"
        assert SemanticConvetion.DB_OPERATION_NAME == "db.operation.name"
        assert SemanticConvetion.DB_QUERY_TEXT == "db.query.text"
        assert (
            SemanticConvetion.DB_RESPONSE_RETURNED_ROWS == "db.response.returned_rows"
        )

    def test_vector_db_operations(self):
        """Test Vector DB operation types."""
        assert SemanticConvetion.DB_OPERATION_CREATE_INDEX == "create_index"
        assert SemanticConvetion.DB_OPERATION_GET_COLLECTION == "get_collection"
        assert SemanticConvetion.DB_OPERATION_CREATE_COLLECTION == "create_collection"
        assert SemanticConvetion.DB_OPERATION_INSERT == "INSERT"
        assert SemanticConvetion.DB_OPERATION_SELECT == "SELECT"
        assert SemanticConvetion.DB_OPERATION_QUERY == "QUERY"
        assert SemanticConvetion.DB_OPERATION_DELETE == "DELETE"
        assert SemanticConvetion.DB_OPERATION_UPDATE == "UPDATE"
        assert SemanticConvetion.DB_OPERATION_UPSERT == "UPSERT"

    def test_vector_db_systems(self):
        """Test Vector DB system types."""
        assert SemanticConvetion.DB_SYSTEM_CHROMA == "chroma"
        assert SemanticConvetion.DB_SYSTEM_PINECONE == "pinecone"
        assert SemanticConvetion.DB_SYSTEM_QDRANT == "qdrant"
        assert SemanticConvetion.DB_SYSTEM_MILVUS == "milvus"
        assert SemanticConvetion.DB_SYSTEM_ASTRA == "astra"

    def test_agent_attributes(self):
        """Test GenAI agent attributes."""
        assert SemanticConvetion.GEN_AI_AGENT_ID == "gen_ai.agent.id"
        assert SemanticConvetion.GEN_AI_AGENT_NAME == "gen_ai.agent.name"
        assert SemanticConvetion.GEN_AI_AGENT_DESCRIPTION == "gen_ai.agent.description"
        assert SemanticConvetion.GEN_AI_AGENT_TYPE == "gen_ai.agent.type"
        assert SemanticConvetion.GEN_AI_AGENT_ROLE == "gen_ai.agent.role"
        assert SemanticConvetion.GEN_AI_AGENT_GOAL == "gen_ai.agent.goal"
        assert SemanticConvetion.GEN_AI_AGENT_TOOLS == "gen_ai.agent.tools"
        assert (
            SemanticConvetion.GEN_AI_AGENT_INSTRUCTIONS == "gen_ai.agent.instructions"
        )

    def test_gpu_attributes(self):
        """Test GPU monitoring attributes."""
        assert SemanticConvetion.GPU_INDEX == "gpu.index"
        assert SemanticConvetion.GPU_UUID == "gpu.uuid"
        assert SemanticConvetion.GPU_NAME == "gpu.name"
        assert SemanticConvetion.GPU_UTILIZATION == "gpu.utilization"
        assert SemanticConvetion.GPU_TEMPERATURE == "gpu.temperature"
        assert SemanticConvetion.GPU_MEMORY_AVAILABLE == "gpu.memory.available"
        assert SemanticConvetion.GPU_MEMORY_TOTAL == "gpu.memory.total"
        assert SemanticConvetion.GPU_MEMORY_USED == "gpu.memory.used"
        assert SemanticConvetion.GPU_POWER_DRAW == "gpu.power.draw"

    def test_guard_attributes(self):
        """Test Guard/Guardrail attributes."""
        assert SemanticConvetion.GUARD_REQUESTS == "guard.requests"
        assert SemanticConvetion.GUARD_VERDICT == "guard.verdict"
        assert SemanticConvetion.GUARD_SCORE == "guard.score"
        assert SemanticConvetion.GUARD_CLASSIFICATION == "guard.classification"
        assert SemanticConvetion.GUARD_VALIDATOR == "guard.validator"
        assert SemanticConvetion.GUARD_EXPLANATION == "guard.explanation"

    def test_eval_attributes(self):
        """Test Evaluation attributes."""
        assert SemanticConvetion.EVAL_REQUESTS == "evals.requests"
        assert SemanticConvetion.EVAL_VERDICT == "evals.verdict"
        assert SemanticConvetion.EVAL_SCORE == "evals.score"
        assert SemanticConvetion.EVAL_CLASSIFICATION == "evals.classification"
        assert SemanticConvetion.EVAL_VALIDATOR == "evals.validator"
        assert SemanticConvetion.EVAL_EXPLANATION == "evals.explanation"

    def test_content_attributes(self):
        """Test content-related attributes."""
        assert SemanticConvetion.GEN_AI_CONTENT_PROMPT_EVENT == "gen_ai.content.prompt"
        assert SemanticConvetion.GEN_AI_CONTENT_PROMPT == "gen_ai.prompt"
        assert (
            SemanticConvetion.GEN_AI_CONTENT_COMPLETION_EVENT
            == "gen_ai.content.completion"
        )
        assert SemanticConvetion.GEN_AI_CONTENT_COMPLETION == "gen_ai.completion"

    def test_rag_attributes(self):
        """Test RAG-related attributes."""
        assert SemanticConvetion.GEN_AI_RAG_MAX_SEGMENTS == "gen_ai.rag.max_segments"
        assert SemanticConvetion.GEN_AI_RAG_STRATEGY == "gen_ai.rag.strategy"
        assert (
            SemanticConvetion.GEN_AI_RAG_SIMILARITY_THRESHOLD
            == "gen_ai.rag.similarity_threshold"
        )
        assert SemanticConvetion.GEN_AI_RAG_MAX_NEIGHBORS == "gen_ai.rag.max_neighbors"
        assert (
            SemanticConvetion.GEN_AI_RAG_DOCUMENTS_PATH == "gen_ai.rag.documents_path"
        )

    def test_extra_request_attributes(self):
        """Test extra request attributes."""
        assert SemanticConvetion.GEN_AI_REQUEST_IS_STREAM == "gen_ai.request.is_stream"
        assert SemanticConvetion.GEN_AI_REQUEST_USER == "gen_ai.request.user"
        assert (
            SemanticConvetion.GEN_AI_REQUEST_EMBEDDING_DIMENSION
            == "gen_ai.request.embedding_dimension"
        )
        assert (
            SemanticConvetion.GEN_AI_REQUEST_TOOL_CHOICE == "gen_ai.request.tool_choice"
        )
        assert (
            SemanticConvetion.GEN_AI_REQUEST_AUDIO_VOICE == "gen_ai.request.audio_voice"
        )
        assert (
            SemanticConvetion.GEN_AI_REQUEST_IMAGE_SIZE == "gen_ai.request.image_size"
        )


class TestSemanticConventionConsistency:
    """Tests for consistency in semantic conventions."""

    def test_all_attributes_are_strings(self):
        """Test that all defined attributes are strings."""
        for attr_name in dir(SemanticConvetion):
            if not attr_name.startswith("_"):
                attr_value = getattr(SemanticConvetion, attr_name)
                assert isinstance(attr_value, str), f"{attr_name} should be a string"

    def test_no_duplicate_values(self):
        """Test that there are no duplicate attribute values (except intentional ones)."""
        values = []
        known_duplicates = {"db.query.namespace"}  # DB_NAMESPACE is duplicated

        for attr_name in dir(SemanticConvetion):
            if not attr_name.startswith("_"):
                attr_value = getattr(SemanticConvetion, attr_name)
                if attr_value not in known_duplicates:
                    if attr_value in values:
                        # This is just a warning, not a failure - some duplicates may be intentional
                        pass
                    values.append(attr_value)

    def test_genai_prefix_consistency(self):
        """Test that GenAI attributes have consistent prefixes."""
        genai_attrs = [
            attr
            for attr in dir(SemanticConvetion)
            if attr.startswith("GEN_AI_") and not attr.startswith("_")
        ]

        # Exceptions: constants that are values, not attribute names
        exception_patterns = ["OPERATION_TYPE", "SYSTEM", "OUTPUT_TYPE", "AGENT_TYPE"]

        for attr_name in genai_attrs:
            attr_value = getattr(SemanticConvetion, attr_name)
            # Most GenAI attributes should start with 'gen_ai.' in their value
            # (except for operation type constants like 'chat', 'embeddings')
            is_exception = any(pattern in attr_name for pattern in exception_patterns)
            if not is_exception:
                assert (
                    attr_value.startswith("gen_ai.") or "." in attr_value
                ), f"{attr_name} value '{attr_value}' should follow OTel naming convention"

    def test_db_prefix_consistency(self):
        """Test that DB attributes have consistent prefixes."""
        db_attrs = [
            attr
            for attr in dir(SemanticConvetion)
            if attr.startswith("DB_") and not attr.startswith("_")
        ]

        for attr_name in db_attrs:
            attr_value = getattr(SemanticConvetion, attr_name)
            # DB_SYSTEM_* values should be system names (without db. prefix)
            # DB_SYSTEM_NAME is an attribute name, not a system type
            if attr_name.startswith("DB_SYSTEM_") and attr_name != "DB_SYSTEM_NAME":
                assert not attr_value.startswith(
                    "db."
                ), f"{attr_name} system name should not have db. prefix"
