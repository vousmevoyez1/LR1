#!/usr/bin/env python3
"""
Sanity check for Qwen3 thought token implementation.

Tests:
1. Thought/final token setup from config
2. Attention mask construction (only when thought+final are enabled)
3. Forward pass aligned with config (thought/final on or off)
4. Query/doc encoding mode (asymmetric or symmetric)
"""

import sys
import os
import argparse
import torch
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, '/data/LR1/src')

from models.qwen3.qwen3_thought_wrapper import Qwen3VLThoughtWrapper


DEFAULT_CONFIG_PATH = "/data/LR1/src/models/qwen3/qwen3/configs/2b/train/inbatch/inbatch.yaml"


def load_reasoning_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model

    num_thought_tokens = int(getattr(model_cfg, "num_thought_tokens", getattr(model_cfg, "mbeir_num_thought_tokens", 0)))
    enable_final_token = bool(getattr(model_cfg, "enable_final_token", False))
    model_name_or_path = getattr(model_cfg, "original_model_name", "/data/Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-2B")
    max_length = int(getattr(model_cfg, "mbeir_max_text_length", 512))
    symmetric_qc_encoding = bool(getattr(model_cfg, "symmetric_query_candidate_encoding", False))

    return {
        "num_thought_tokens": num_thought_tokens,
        "enable_final_token": enable_final_token,
        "symmetric_query_candidate_encoding": symmetric_qc_encoding,
        "model_name_or_path": model_name_or_path,
        "max_length": max_length,
        "config_path": config_path,
    }


def build_model(settings):
    model = Qwen3VLThoughtWrapper(
        model_name_or_path=settings["model_name_or_path"],
        max_length=settings["max_length"],
    )
    model.setup_thought_tokens(
        settings["num_thought_tokens"],
        enable_final_token=settings["enable_final_token"],
    )
    return model


def test_thought_token_setup(settings):
    """Test 1: Thought/final token setup from config."""
    print("\n" + "="*80)
    print("Test 1: Thought/Final Token Setup From Config")
    print("="*80)

    model = build_model(settings)
    num_thought_tokens = settings["num_thought_tokens"]
    enable_final_token = settings["enable_final_token"]

    # Verify thought token IDs
    assert model._thought_token_ids is not None, "Thought token IDs not set"
    assert len(model._thought_token_ids) == num_thought_tokens, f"Expected {num_thought_tokens} thought tokens"

    # Verify final token setting
    if enable_final_token:
        assert model._final_token_id is not None, "Final token ID not set while enable_final_token=True"
    else:
        assert model._final_token_id is None, "Final token ID should be None while enable_final_token=False"

    # Verify uniqueness
    all_ids = list(model._thought_token_ids)
    if model._final_token_id is not None:
        all_ids.append(model._final_token_id)
    assert len(set(all_ids)) == len(all_ids), "Token IDs are not unique!"

    print(f"✓ Config: {settings['config_path']}")
    print(f"✓ num_thought_tokens={num_thought_tokens}, enable_final_token={enable_final_token}")
    print(f"✓ Thought token IDs: {model._thought_token_ids}")
    print(f"✓ Final token ID: {model._final_token_id}")
    print(f"✓ All IDs are unique")

    return model


def test_attention_mask(settings):
    """Test 2: Attention mask construction."""
    print("\n" + "="*80)
    print("Test 2: Attention Mask Construction")
    print("="*80)

    num_thought_tokens = settings["num_thought_tokens"]
    enable_final_token = settings["enable_final_token"]

    if num_thought_tokens <= 0 or not enable_final_token:
        print("- Skipped: attention-mask restriction test requires num_thought_tokens>0 and enable_final_token=True")
        return None

    model = build_model(settings)

    # Create dummy input_ids with thought tokens at the end
    batch_size = 2
    seq_len = 20
    device = model.model.device

    # Simulate: [query tokens] [thought_1..K] [final]
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    start = seq_len - (num_thought_tokens + 1)
    for i in range(num_thought_tokens):
        input_ids[:, start + i] = model._thought_token_ids[i]
    input_ids[:, 19] = model._final_token_id

    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # Build reasoning attention mask
    reasoning_mask = model._build_reasoning_attention_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_thought_tokens=num_thought_tokens,
    )

    # Verify shape
    assert reasoning_mask.shape == (batch_size, 1, seq_len, seq_len), \
        f"Expected shape ({batch_size}, 1, {seq_len}, {seq_len}), got {reasoning_mask.shape}"

    # Verify final token can only attend to thought tokens + itself
    final_pos = 19
    for b in range(batch_size):
        final_row = reasoning_mask[b, 0, final_pos, :]

        # Check that original query tokens are masked
        for pos in range(start):
            assert final_row[pos] < 0, f"Final token should not attend to position {pos}"

        # Check that thought tokens and final are allowed
        for pos in range(start, 20):
            assert final_row[pos] == 0.0, f"Final token should attend to position {pos}"

    print(f"✓ Attention mask shape: {reasoning_mask.shape}")
    print(f"✓ Final token correctly restricted to thought tokens + itself")

    return model


def test_forward_pass(settings):
    """Test 3: Forward pass aligned with config."""
    print("\n" + "="*80)
    print("Test 3: Forward Pass (Config-Aligned Thought/Final)")
    print("="*80)

    num_thought_tokens = settings["num_thought_tokens"]
    enable_final_token = settings["enable_final_token"]

    model = build_model(settings)
    model.eval()

    # Test text-only input
    texts = ["This is a test query."]

    with torch.no_grad():
        # Preprocess with thought tokens
        inputs = model._preprocess_inputs(
            texts=texts,
            num_thought_tokens=num_thought_tokens,
            use_final_token=enable_final_token,
        )
        inputs = {k: v.to(model.model.device) for k, v in inputs.items()}

        # Forward pass
        output = model._forward_embeddings(
            num_thought_tokens=num_thought_tokens,
            use_final_token=enable_final_token,
            **inputs,
        )

        # Verify output
        assert output.single_vec_emb is not None, "No embedding output"
        assert output.single_vec_emb.shape[0] == 1, "Batch size mismatch"
        assert output.single_vec_emb.shape[1] == model.model.config.text_config.hidden_size, "Embedding dim mismatch"

        # Verify embedding is normalized
        norm = torch.norm(output.single_vec_emb, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), "Embedding not normalized"

        # Verify final token existence based on config
        if enable_final_token and model._final_token_id is not None:
            has_final = (inputs["input_ids"] == model._final_token_id).any().item()
            assert has_final, "Expected <final> token in query input, but not found"
        elif model._final_token_id is not None:
            has_final = (inputs["input_ids"] == model._final_token_id).any().item()
            assert not has_final, "Did not expect <final> token in query input"

        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {output.single_vec_emb.shape}")
        print(f"✓ Config used: TT={num_thought_tokens}, FT={enable_final_token}")
        print(f"✓ Embedding is normalized (norm={norm.item():.6f})")

    return model


def test_query_candidate_encoding_mode(settings):
    """Test 4: Query/doc encoding mode (asymmetric/symmetric)."""
    print("\n" + "="*80)
    print("Test 4: Query/Doc Encoding Mode")
    print("="*80)

    num_thought_tokens = settings["num_thought_tokens"]
    enable_final_token = settings["enable_final_token"]
    symmetric_qc_encoding = settings["symmetric_query_candidate_encoding"]

    model = build_model(settings)
    model.mbeir_num_thought_tokens = num_thought_tokens
    model.enable_final_token = enable_final_token
    model.mbeir_symmetric_encoding = symmetric_qc_encoding
    model.eval()

    # Create dummy MBEIR batch (query)
    from models.qwen3.qwen3_thought_wrapper import RawTextBatch

    batch_size = 2
    device = model.model.device

    query_batch = {
        "qid_list": ["q1", "q2"],
        "txt_batched": RawTextBatch(["query 1", "query 2"]),
        "image_batched": torch.zeros(batch_size, 3, 224, 224),
        "txt_mask_batched": torch.ones(batch_size),
        "image_mask_batched": torch.zeros(batch_size),
    }

    doc_batch = {
        "did_list": ["d1", "d2"],
        "txt_batched": RawTextBatch(["doc 1", "doc 2"]),
        "image_batched": torch.zeros(batch_size, 3, 224, 224),
        "txt_mask_batched": torch.ones(batch_size),
        "image_mask_batched": torch.zeros(batch_size),
    }

    with torch.no_grad():
        records = {"query": [], "doc": []}
        current_phase = {"name": "unknown"}

        orig_preprocess = model._preprocess_inputs
        orig_forward = model._forward_embeddings

        def tracked_preprocess(*args, **kwargs):
            records[current_phase["name"]].append((
                "preprocess",
                kwargs.get("num_thought_tokens", 0),
                kwargs.get("use_final_token", None),
            ))
            return orig_preprocess(*args, **kwargs)

        def tracked_forward(*args, **kwargs):
            records[current_phase["name"]].append((
                "forward",
                kwargs.get("num_thought_tokens", 0),
                kwargs.get("use_final_token", False),
            ))
            return orig_forward(*args, **kwargs)

        model._preprocess_inputs = tracked_preprocess
        model._forward_embeddings = tracked_forward

        # Encode query (should use thought tokens)
        current_phase["name"] = "query"
        query_embs, query_ids = model.encode_mbeir_batch(query_batch)

        # Encode doc (should NOT use thought tokens)
        current_phase["name"] = "doc"
        doc_embs, doc_ids = model.encode_mbeir_batch(doc_batch)

        # Restore original methods
        model._preprocess_inputs = orig_preprocess
        model._forward_embeddings = orig_forward

        # Verify shapes
        assert query_embs.shape == (batch_size, model.model.config.text_config.hidden_size), \
            f"Query embedding shape mismatch: {query_embs.shape}"
        assert doc_embs.shape == (batch_size, model.model.config.text_config.hidden_size), \
            f"Doc embedding shape mismatch: {doc_embs.shape}"

        # Verify IDs
        assert query_ids == ["q1", "q2"], "Query IDs mismatch"
        assert doc_ids == ["d1", "d2"], "Doc IDs mismatch"

        # Verify token routing for query/doc
        assert records["query"], "No query-side preprocess/forward calls captured"
        assert records["doc"], "No doc-side preprocess/forward calls captured"

        for stage, tt, ft in records["query"]:
            assert tt == num_thought_tokens, f"Query {stage} should use TT={num_thought_tokens}, got {tt}"
            assert bool(ft) == enable_final_token, f"Query {stage} should use FT={enable_final_token}, got {ft}"

        expected_doc_tt = num_thought_tokens if symmetric_qc_encoding else 0
        expected_doc_ft = enable_final_token if symmetric_qc_encoding else False

        for stage, tt, ft in records["doc"]:
            assert tt == expected_doc_tt, (
                f"Doc {stage} should use TT={expected_doc_tt}, got {tt}"
            )
            assert bool(ft) is bool(expected_doc_ft), (
                f"Doc {stage} should use FT={expected_doc_ft}, got {ft}"
            )

        print(f"✓ Query encoding successful: {query_embs.shape}")
        print(f"✓ Doc encoding successful: {doc_embs.shape}")
        print(f"✓ Query tokens: TT={num_thought_tokens}, FT={enable_final_token}")
        print(f"✓ Candidate tokens: TT={expected_doc_tt}, FT={expected_doc_ft}")
        mode_name = "symmetric" if symmetric_qc_encoding else "asymmetric"
        print(f"✓ {mode_name} encoding mode works correctly")

    return model


def main():
    parser = argparse.ArgumentParser(description="Qwen3 thought/final token sanity checks")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config. Test uses model.num_thought_tokens + model.enable_final_token",
    )
    args = parser.parse_args()

    settings = load_reasoning_config(args.config)

    print("\n" + "="*80)
    print("Qwen3 Thought Token Implementation - Sanity Checks")
    print("="*80)
    print(f"Config: {settings['config_path']}")
    print(
        f"TT={settings['num_thought_tokens']}, FT={settings['enable_final_token']}, "
        f"SYM={settings['symmetric_query_candidate_encoding']}"
    )

    try:
        # Run all tests
        test_thought_token_setup(settings)
        test_attention_mask(settings)
        test_forward_pass(settings)
        test_query_candidate_encoding_mode(settings)

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        return 0

    except Exception as e:
        print("\n" + "="*80)
        print(f"✗ TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
