"""Tests for the ONNX Runtime embeddings provider."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hindsight_api.engine.embeddings import OnnxEmbeddings, create_embeddings_from_env


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        self.calls.append(
            {
                "texts": texts,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        batch = len(texts)
        return {
            "input_ids": np.ones((batch, 3), dtype=np.int64),
            "attention_mask": np.array([[1, 1, 0]] * batch, dtype=np.int64),
            "token_type_ids": np.zeros((batch, 3), dtype=np.int64),
        }


class FakeOnnxSession:
    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, output_names, inputs):
        batch = inputs["input_ids"].shape[0]
        # Last token is masked out. Mean pooling should average first two tokens:
        # ([3, 4] + [0, 0]) / 2 = [1.5, 2.0], then normalize to [0.6, 0.8].
        token_embeddings = np.array([[[3.0, 4.0], [0.0, 0.0], [100.0, 100.0]]] * batch, dtype=np.float32)
        return [token_embeddings]


class RowNumberingOnnxSession:
    """Returns a distinct embedding per row so batch splitting and ordering are verifiable.

    Row *n* (counted across the whole run, not per batch) gets token embeddings
    ``[[n, 0], [0, 0], [100, 100]]``; with ``FakeTokenizer``'s ``[1, 1, 0]`` mask,
    mean pooling yields ``[n / 2, 0]``.
    """

    def __init__(self):
        self.batch_sizes: list[int] = []
        self._next_row = 1.0

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, output_names, inputs):
        batch = inputs["input_ids"].shape[0]
        self.batch_sizes.append(batch)
        rows = []
        for _ in range(batch):
            rows.append([[self._next_row, 0.0], [0.0, 0.0], [100.0, 100.0]])
            self._next_row += 1.0
        return [np.array(rows, dtype=np.float32)]


class FakePooledOnnxSession:
    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, output_names, inputs):
        batch = inputs["input_ids"].shape[0]
        assert output_names == ["sentence_embedding"]
        return [np.array([[3.0, 4.0]] * batch, dtype=np.float32)]


def test_onnx_embeddings_mean_pooling_normalizes_and_filters_inputs():
    emb = OnnxEmbeddings(model_id="intfloat/multilingual-e5-small", dimensions=2, max_tokens=17)
    emb._tokenizer = FakeTokenizer()
    emb._session = FakeOnnxSession()
    emb._dimension = 2

    result = emb.encode(["hello"])

    assert result == [pytest.approx([0.6, 0.8])]
    assert emb._tokenizer.calls[-1]["max_length"] == 17


def test_onnx_embeddings_cls_pooling_and_normalize_false():
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        dimensions=2,
        pooling="cls",
        normalize=False,
    )
    emb._tokenizer = FakeTokenizer()
    emb._session = FakeOnnxSession()
    emb._dimension = 2

    result = emb.encode(["hello"])

    assert result == [pytest.approx([3.0, 4.0])]


def test_onnx_embeddings_output_name_uses_pre_pooled_2d_output():
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        dimensions=2,
        output_name="sentence_embedding",
    )
    emb._tokenizer = FakeTokenizer()
    emb._session = FakePooledOnnxSession()
    emb._dimension = 2

    result = emb.encode(["hello"])

    assert result == [pytest.approx([0.6, 0.8])]


def test_onnx_embeddings_rejects_invalid_pooling_before_initialize():
    with pytest.raises(ValueError, match="pooling"):
        OnnxEmbeddings(model_id="intfloat/multilingual-e5-small", pooling="max")


def test_onnx_embeddings_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        OnnxEmbeddings(model_id="intfloat/multilingual-e5-small", batch_size=0)


def test_onnx_embeddings_splits_into_batches_preserving_order():
    """Texts must be encoded in ``batch_size`` chunks, in order, with none dropped.

    An unbatched pass pads every row to the longest sequence in the whole list,
    so a large retain allocates one huge tensor and ratchets ONNX Runtime's RSS
    up permanently.
    """
    session = RowNumberingOnnxSession()
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        dimensions=2,
        normalize=False,
        batch_size=2,
    )
    emb._tokenizer = FakeTokenizer()
    emb._session = session
    emb._dimension = 2

    result = emb.encode(["a", "b", "c", "d", "e"])

    assert session.batch_sizes == [2, 2, 1]
    assert emb._tokenizer.calls[0]["texts"] == ["a", "b"]
    assert emb._tokenizer.calls[-1]["texts"] == ["e"]
    assert result == [
        pytest.approx([0.5, 0.0]),
        pytest.approx([1.0, 0.0]),
        pytest.approx([1.5, 0.0]),
        pytest.approx([2.0, 0.0]),
        pytest.approx([2.5, 0.0]),
    ]


def test_onnx_embeddings_batching_matches_single_pass_result():
    """Batching must not change the vectors — pooling masks padding out either way."""

    def encode_with(batch_size: int) -> list[list[float]]:
        emb = OnnxEmbeddings(
            model_id="intfloat/multilingual-e5-small",
            dimensions=2,
            batch_size=batch_size,
        )
        emb._tokenizer = FakeTokenizer()
        emb._session = FakeOnnxSession()
        emb._dimension = 2
        return emb.encode(["a", "b", "c"])

    assert encode_with(1) == encode_with(1000)


def test_onnx_embeddings_warns_when_local_model_path_has_no_tokenizer(caplog):
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        model_path="/models/custom/onnx/model.onnx",
    )

    assert emb.tokenizer_name_or_path == "intfloat/multilingual-e5-small"
    assert "model_path is set without tokenizer_name_or_path" in caplog.text


def test_onnx_embeddings_query_and_document_prefixes_are_asymmetric():
    tokenizer = FakeTokenizer()
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        dimensions=2,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    emb._tokenizer = tokenizer
    emb._session = FakeOnnxSession()
    emb._dimension = 2

    emb.encode_query(["weather"])
    emb.encode_documents(["weather"])

    assert tokenizer.calls[0]["texts"] == ["query: weather"]
    assert tokenizer.calls[1]["texts"] == ["passage: weather"]


@pytest.mark.asyncio
async def test_onnx_embeddings_dimension_mismatch_raises_value_error():
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        model_path="/models/e5/onnx/model.onnx",
        tokenizer_name_or_path="/models/e5",
        dimensions=3,
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock(return_value=FakeTokenizer()))
    )
    fake_onnxruntime = SimpleNamespace(
        InferenceSession=MagicMock(return_value=FakeOnnxSession()),
        SessionOptions=SimpleNamespace,
    )

    with patch.dict(sys.modules, {"transformers": fake_transformers, "onnxruntime": fake_onnxruntime}):
        with pytest.raises(ValueError, match="does not match model output"):
            await emb.initialize()


@pytest.mark.asyncio
async def test_onnx_embeddings_downloads_external_data_sidecar_when_needed():
    emb = OnnxEmbeddings(model_id="BAAI/bge-m3", onnx_file="onnx/model.onnx")
    download = MagicMock(return_value="/hf/bge-m3")
    session = MagicMock(return_value=FakeOnnxSession())
    fake_hf = SimpleNamespace(snapshot_download=download)
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock(return_value=FakeTokenizer()))
    )
    fake_onnxruntime = SimpleNamespace(InferenceSession=session, SessionOptions=SimpleNamespace)

    with patch.dict(
        sys.modules,
        {
            "huggingface_hub": fake_hf,
            "transformers": fake_transformers,
            "onnxruntime": fake_onnxruntime,
        },
    ):
        await emb.initialize()

    download.assert_called_once_with(
        repo_id="BAAI/bge-m3",
        allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data"],
    )
    assert session.call_args.args == ("/hf/bge-m3/onnx/model.onnx",)
    assert session.call_args.kwargs["providers"] == ["CPUExecutionProvider"]


@pytest.mark.asyncio
async def test_onnx_embeddings_disables_cpu_mem_arena():
    """The ONNX CPU arena never returns memory to the OS, so it must stay off.

    With it enabled, one oversized batch raises RSS for the life of the process
    — the reranker disables it for the same reason (see ``cross_encoder.py``).
    """
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        model_path="/models/e5/onnx/model.onnx",
        tokenizer_name_or_path="/models/e5",
        dimensions=2,
    )
    session = MagicMock(return_value=FakeOnnxSession())
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock(return_value=FakeTokenizer()))
    )
    fake_onnxruntime = SimpleNamespace(InferenceSession=session, SessionOptions=SimpleNamespace)

    with patch.dict(sys.modules, {"transformers": fake_transformers, "onnxruntime": fake_onnxruntime}):
        await emb.initialize()

    assert session.call_args.kwargs["sess_options"].enable_cpu_mem_arena is False


def test_create_embeddings_from_env_supports_onnx_provider():
    mock_config = MagicMock()
    mock_config.embeddings_provider = "onnx"
    mock_config.embeddings_onnx_model_id = "intfloat/multilingual-e5-small"
    mock_config.embeddings_onnx_model_path = "/models/e5/onnx/model.onnx"
    mock_config.embeddings_onnx_tokenizer_name_or_path = "/models/e5"
    mock_config.embeddings_onnx_file = "onnx/model.onnx"
    mock_config.embeddings_onnx_dimensions = 384
    mock_config.embeddings_onnx_max_tokens = 512
    mock_config.embeddings_onnx_pooling = "mean"
    mock_config.embeddings_onnx_normalize = True
    mock_config.embeddings_onnx_query_prefix = "query: "
    mock_config.embeddings_onnx_passage_prefix = "passage: "
    mock_config.embeddings_onnx_output_name = None

    with patch("hindsight_api.config.get_config", return_value=mock_config):
        emb = create_embeddings_from_env()

    assert isinstance(emb, OnnxEmbeddings)
    assert emb.provider_name == "onnx"
    assert emb.model_id == "intfloat/multilingual-e5-small"
    assert emb.model_path == "/models/e5/onnx/model.onnx"
    assert emb.tokenizer_name_or_path == "/models/e5"
    assert emb.dimension == 384
