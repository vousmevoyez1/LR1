"""
Utility functions for retrieval experiments on MBEIR.
"""

# Standard Library imports
import os
import random

# Third-party imports
import numpy as np
import torch


def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            if int(relevance_score) > 0:  # Assuming only positive relevance scores indicate relevant documents
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Loaded {len(qrel)} queries from {filename}")
    print(f"Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}")
    return qrel, qid_to_taskid


# TODO: Write a hashed id to unhased id converter.
def load_runfile(filename, load_task_id=False):
    run_results = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            qid = parts[0]
            if qid not in run_results:
                run_results[qid] = []
            if load_task_id:
                qid, _, did, rank, score, run_id, task_id = parts
                run_results[qid].append(
                    {
                        "did": did,
                        "rank": int(rank),
                        "score": float(score),
                        "task_id": task_id,
                    }
                )
            else:
                parts = parts[:6]
                qid, _, did, rank, score, run_id = parts
                run_results[qid].append(
                    {
                        "did": did,
                        "rank": int(rank),
                        "score": float(score),
                    }
                )
    print(f"Loaded results for {len(run_results)} queries from {filename}")
    return run_results


def build_model_from_config(config):
    model_name = config.model.name

    if model_name == "CLIPScoreFusion":
        from models.uniir_clip.clip_scorefusion.clip_sf import CLIPScoreFusion

        # initialize CLIPScoreFusion model
        model_config = config.model
        uniir_dir = config.uniir_dir
        download_root = os.path.join(uniir_dir, model_config.pretrained_clip_model_dir)
        print(f"Downloading CLIP model to {download_root}...")
        model = CLIPScoreFusion(
            model_name=model_config.clip_vision_model_name,
            download_root=download_root,
        )
        model.float()
        # The origial CLIP was in fp16 so we need to convert it to fp32

        # Load model from checkpoint
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading CLIPScoreFusion checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)["model"])

    elif model_name == "CLIPFeatureFusion":
        from models.uniir_clip.clip_featurefusion.clip_ff import CLIPFeatureFusion

        # initialize CLIPFeatureFusion model
        model_config = config.model
        uniir_dir = config.uniir_dir
        download_root = os.path.join(uniir_dir, model_config.pretrained_clip_model_dir)
        print(f"Downloading CLIP model to {download_root}...")
        model = CLIPFeatureFusion(
            model_name=model_config.clip_vision_model_name,
            download_root=download_root,
        )
        model.float()
        # The origial CLIP was in fp16 so we need to convert it to fp32

        # Load model from checkpoint
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading CLIPFeatureFusion checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)["model"])

    elif model_name == "BLIPScoreFusion":
        from models.uniir_blip.blip_scorefusion.blip_sf import BLIPScoreFusion

        model_config = config.model
        model = BLIPScoreFusion(
            med_config=os.path.join("../models/uniir_blip", "backbone/configs/med_config.json"),
            image_size=model_config.image_size,
            vit=model_config.vit,
            vit_grad_ckpt=model_config.vit_grad_ckpt,
            vit_ckpt_layer=model_config.vit_ckpt_layer,
            embed_dim=model_config.embed_dim,
            queue_size=model_config.queue_size,
            config=model_config,
        )
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading BLIPScoreFusion checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)["model"])

    elif model_name == "BLIPFeatureFusion":
        from models.uniir_blip.blip_featurefusion.blip_ff import BLIPFeatureFusion

        model_config = config.model
        model = BLIPFeatureFusion(
            med_config=os.path.join("../models/uniir_blip", "backbone/configs/med_config.json"),
            image_size=model_config.image_size,
            vit=model_config.vit,
            vit_grad_ckpt=model_config.vit_grad_ckpt,
            vit_ckpt_layer=model_config.vit_ckpt_layer,
            embed_dim=model_config.embed_dim,
            queue_size=model_config.queue_size,
            config=model_config,
        )
        ckpt_config = model_config.ckpt_config
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading BLIPFeatureFusion checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)["model"])
    elif model_name == "JinaEmbeddingsV4Model":
        ##需要修改路径
        from models.jina_v4.jina_v4.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model

        model_config = config.model
        # Initialize the model using the custom from_pretrained method
        model = JinaEmbeddingsV4Model.from_pretrained(
            model_config.original_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            ### 原来是flash attention，这里改回sdpa，懒得下载flash attention了
            attn_implementation="sdpa",
        )

        # Set MBEIR-specific attributes
        model.mbeir_task_label = getattr(model_config, 'mbeir_task_label', 'retrieval')
        model.mbeir_image_size = tuple(map(int, config.data_config.image_size.split(",")))
        model.mbeir_max_text_length = getattr(model_config, 'mbeir_max_text_length', 512)
        # Set reason_steps for implicit reasoning in embedding space
        model.mbeir_reason_steps = getattr(model_config, 'mbeir_reason_steps', 0)
        # Query/Candidate encoding mode in MBEIR embedding:
        # False (default): query uses reason_steps, candidate uses 0
        # True: both query and candidate use reason_steps
        model.mbeir_symmetric_reasoning = getattr(model_config, 'symmetric_query_candidate_encoding', False)
        # Set the task for LoRA adapter selection
        model.task = model.mbeir_task_label

    elif model_name == "JinaEmbeddingsV4aModel":
        # jina_v4a: 新的多步推理方式
        from models.jina_v4a.jina_v4a.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model

        model_config = config.model
        # Initialize the model using the custom from_pretrained method
        model = JinaEmbeddingsV4Model.from_pretrained(
            model_config.original_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

        # Set MBEIR-specific attributes
        model.mbeir_task_label = getattr(model_config, 'mbeir_task_label', 'retrieval')
        model.mbeir_image_size = tuple(map(int, config.data_config.image_size.split(",")))
        model.mbeir_max_text_length = getattr(model_config, 'mbeir_max_text_length', 512)
        # Set reason_steps for implicit reasoning in embedding space
        model.mbeir_reason_steps = getattr(model_config, 'mbeir_reason_steps', 0)
        # Set the task for LoRA adapter selection
        model.task = model.mbeir_task_label

    elif model_name == "JinaEmbeddingsV4opModel":
        # jina_v4o: Thought Token (思维锚点) 推理方式
        from models.jina_v4op.jina_v4op.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model

        model_config = config.model
        # Initialize the model using the custom from_pretrained method
        model = JinaEmbeddingsV4Model.from_pretrained(
            model_config.original_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

        # Set MBEIR-specific attributes
        model.mbeir_task_label = getattr(model_config, 'mbeir_task_label', 'retrieval')
        model.mbeir_image_size = tuple(map(int, config.data_config.image_size.split(",")))
        model.mbeir_max_text_length = getattr(model_config, 'mbeir_max_text_length', 512)
        # Set num_thought_tokens for thought token reasoning
        model.mbeir_num_thought_tokens = getattr(model_config, 'mbeir_num_thought_tokens', 0)
        # Query/Candidate encoding mode for evaluation/testing
        model.mbeir_symmetric_encoding = getattr(model_config, 'symmetric_query_candidate_encoding', False)
        # Final embedding mode for thought tokens: last | mean_all_thought_tokens
        model.thought_pooling_mode = getattr(model_config, 'thought_token_pooling_mode', 'last')
        model.mbeir_thought_pooling_mode = model.thought_pooling_mode
        # Setup thought tokens if specified
        # Note: skip_init=True because we'll load embeddings from checkpoint later
        # The actual embeddings will be loaded by load_lora_checkpoint() in mbeir_embedder.py
        if model.mbeir_num_thought_tokens > 0:
            model.setup_thought_tokens(model.mbeir_num_thought_tokens, skip_init=True)
        # Set the task for LoRA adapter selection
        model.task = model.mbeir_task_label
    elif model_name == "JinaEmbeddingsV4oModel":
        # jina_v4o: Thought Token (思维锚点) 推理方式
        from models.jina_v4o.jina_v4o.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model

        model_config = config.model
        # Initialize the model using the custom from_pretrained method
        model = JinaEmbeddingsV4Model.from_pretrained(
            model_config.original_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

        # Set MBEIR-specific attributes
        model.mbeir_task_label = getattr(model_config, 'mbeir_task_label', 'retrieval')
        model.mbeir_image_size = tuple(map(int, config.data_config.image_size.split(",")))
        model.mbeir_max_text_length = getattr(model_config, 'mbeir_max_text_length', 512)
        # Set num_thought_tokens for thought token reasoning
        model.mbeir_num_thought_tokens = getattr(model_config, 'mbeir_num_thought_tokens', 0)
        # Query/Candidate encoding mode for evaluation/testing
        model.mbeir_symmetric_encoding = getattr(model_config, 'symmetric_query_candidate_encoding', False)
        # Final embedding mode for thought tokens: last | mean_all_thought_tokens
        model.thought_pooling_mode = getattr(model_config, 'thought_token_pooling_mode', 'last')
        model.mbeir_thought_pooling_mode = model.thought_pooling_mode
        # Setup thought tokens if specified
        # Note: skip_init=True because we'll load embeddings from checkpoint later
        # The actual embeddings will be loaded by load_lora_checkpoint() in mbeir_embedder.py
        if model.mbeir_num_thought_tokens > 0:
            model.setup_thought_tokens(model.mbeir_num_thought_tokens, skip_init=True)
        # Set the task for LoRA adapter selection
        model.task = model.mbeir_task_label
    elif model_name == "JinaEmbeddingsV4oModel":
        # jina_v4o: Thought Token (思维锚点) 推理方式
        from models.jina_v4oaux.jina_v4oaux.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model

        model_config = config.model
        # Initialize the model using the custom from_pretrained method
        model = JinaEmbeddingsV4Model.from_pretrained(
            model_config.original_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

        # Set MBEIR-specific attributes
        model.mbeir_task_label = getattr(model_config, 'mbeir_task_label', 'retrieval')
        model.mbeir_image_size = tuple(map(int, config.data_config.image_size.split(",")))
        model.mbeir_max_text_length = getattr(model_config, 'mbeir_max_text_length', 512)
        # Set num_thought_tokens for thought token reasoning
        model.mbeir_num_thought_tokens = getattr(model_config, 'mbeir_num_thought_tokens', 0)
        # Query/Candidate encoding mode for evaluation/testing
        model.mbeir_symmetric_encoding = getattr(model_config, 'symmetric_query_candidate_encoding', False)
        # Final embedding mode for thought tokens: last | mean_all_thought_tokens
        model.thought_pooling_mode = getattr(model_config, 'thought_token_pooling_mode', 'last')
        model.mbeir_thought_pooling_mode = model.thought_pooling_mode
        # Setup thought tokens if specified
        # Note: skip_init=True because we'll load embeddings from checkpoint later
        # The actual embeddings will be loaded by load_lora_checkpoint() in mbeir_embedder.py
        if model.mbeir_num_thought_tokens > 0:
            model.setup_thought_tokens(model.mbeir_num_thought_tokens, skip_init=True)
        # Set the task for LoRA adapter selection
        model.task = model.mbeir_task_label
    elif model_name == "JinaEmbeddingsV4oauxaModel":
        # jina_v4o: Thought Token (思维锚点) 推理方式
        from models.jina_v4oauxa.jina_v4oauxa.modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model

        model_config = config.model
        # Initialize the model using the custom from_pretrained method
        model = JinaEmbeddingsV4Model.from_pretrained(
            model_config.original_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

        # Set MBEIR-specific attributes
        model.mbeir_task_label = getattr(model_config, 'mbeir_task_label', 'retrieval')
        model.mbeir_image_size = tuple(map(int, config.data_config.image_size.split(",")))
        model.mbeir_max_text_length = getattr(model_config, 'mbeir_max_text_length', 512)
        # Set num_thought_tokens for thought token reasoning
        model.mbeir_num_thought_tokens = getattr(model_config, 'mbeir_num_thought_tokens', 0)
        # Query/Candidate encoding mode for evaluation/testing
        model.mbeir_symmetric_encoding = getattr(model_config, 'symmetric_query_candidate_encoding', False)
        # Final embedding mode for thought tokens: last | mean_all_thought_tokens
        model.thought_pooling_mode = getattr(model_config, 'thought_token_pooling_mode', 'last')
        model.mbeir_thought_pooling_mode = model.thought_pooling_mode
        # Setup thought tokens if specified
        # Note: skip_init=True because we'll load embeddings from checkpoint later
        # The actual embeddings will be loaded by load_lora_checkpoint() in mbeir_embedder.py
        if model.mbeir_num_thought_tokens > 0:
            model.setup_thought_tokens(model.mbeir_num_thought_tokens, skip_init=True)
        # Set the task for LoRA adapter selection
        model.task = model.mbeir_task_label
    elif model_name == "Qwen3VLEmbedder":
        # Import Qwen3VL thought wrapper (supports thought tokens)
        from models.qwen3a.qwen3_thought_wrapper import Qwen3VLThoughtWrapper

        model_config = config.model
        # Initialize the Qwen3VL thought wrapper
        model = Qwen3VLThoughtWrapper(
            model_name_or_path=model_config.original_model_name,
            max_length=getattr(model_config, 'mbeir_max_text_length', 8192),
            torch_dtype=torch.bfloat16,
            # Must use sdpa (not flash_attention_2): the 4D reasoning attention mask
            # used by thought tokens is incompatible with FlashAttention2.
            attn_implementation=getattr(model_config, "attn_implementation", "sdpa"),
        )

        # Set MBEIR-specific attributes
        model.mbeir_task_label = getattr(model_config, 'mbeir_task_label', 'retrieval')
        model.mbeir_image_size = tuple(map(int, config.data_config.image_size.split(",")))
        model.mbeir_max_text_length = getattr(model_config, 'mbeir_max_text_length', 8192)
        model.mbeir_num_thought_tokens = getattr(model_config, 'mbeir_num_thought_tokens', 0)
        model.enable_final_token = getattr(model_config, 'enable_final_token', False)
        model.mbeir_symmetric_encoding = getattr(model_config, 'symmetric_query_candidate_encoding', False)
        model.task = model.mbeir_task_label

        # Setup thought tokens if specified
        # Note: skip_init=True because we'll load embeddings from checkpoint later
        # The actual embeddings will be loaded by load_lora_checkpoint() in mbeir_embedder.py
        if model.mbeir_num_thought_tokens > 0 or model.enable_final_token:
            semantic_init_token = getattr(model_config, 'semantic_init_token', '.')
            skip_init = getattr(model_config, 'skip_thought_init', True)  # Use skip_init=True during eval
            model.setup_thought_tokens(
                model.mbeir_num_thought_tokens,
                semantic_init_token=semantic_init_token,
                skip_init=skip_init,
                enable_final_token=model.enable_final_token
            )
            if model.mbeir_num_thought_tokens > 0:
                print(f"✓ Qwen3 model initialized with {model.mbeir_num_thought_tokens} thought tokens")
            if model.enable_final_token:
                print(f"✓ Qwen3 model initialized with final token")
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented.")
        # Notes: Add other models here
    return model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
