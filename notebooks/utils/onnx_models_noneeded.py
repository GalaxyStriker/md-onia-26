import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

_TOKENIZER = None
_ORT_SESSION = None

def init_model(model_id="Snowflake/snowflake-arctic-embed-l-v2.0", onnx_filename="onnx/model_int8.onnx"):
    global _TOKENIZER, _ORT_SESSION
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    
    if _ORT_SESSION is None:
        model_path = hf_hub_download(repo_id=model_id, filename=onnx_filename)
        
        # Try to download the external data file which ONNX needs for large/quantized models
        try:
            hf_hub_download(repo_id=model_id, filename=onnx_filename + "_data")
        except Exception:
            pass  # It's fine if it doesn't exist for other models
            
        # Enforce CPU execution provider as requested
        providers = ['CPUExecutionProvider']
        _ORT_SESSION = ort.InferenceSession(model_path, providers=providers)

def get_token_level_embeddings(text: str) -> np.ndarray:
    """
    Return token-level embeddings for one text.

    Expected:
        np.ndarray with shape [num_tokens, dim], dtype float32/float16.
    """
    init_model()
    
    # Tokenize text
    inputs = _TOKENIZER(text, return_tensors="np", padding=False, truncation=True)
    
    # Get required ONNX inputs
    input_names = [i.name for i in _ORT_SESSION.get_inputs()]
    
    # Check expected shape to handle rank mismatches (e.g. 1D vs 2D)
    expected_ranks = {i.name: len(i.shape) for i in _ORT_SESSION.get_inputs() if i.shape is not None}
    
    # Prepare inputs for ONNX session based strictly on what's requested
    ort_inputs = {}
    
    def format_input(name, arr):
        # Flatten if the ONNX model expects a 1D array
        if expected_ranks.get(name, 2) == 1:
            return arr.flatten()
        return arr
    
    if "input_ids" in input_names:
        input_ids = inputs["input_ids"].astype(np.int64)
        # Prevent out-of-bound token IDs (e.g. added special tokens missing from ONNX export)
        if hasattr(_TOKENIZER, "vocab_size"):
            input_ids = np.clip(input_ids, 0, _TOKENIZER.vocab_size - 1)
        ort_inputs["input_ids"] = format_input("input_ids", input_ids)
        
    if "attention_mask" in input_names:
        ort_inputs["attention_mask"] = format_input("attention_mask", inputs["attention_mask"].astype(np.int64))
    
    if "token_type_ids" in input_names and "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = format_input("token_type_ids", inputs["token_type_ids"].astype(np.int64))
        
    if "offsets" in input_names:
        # Offsets usually mark the start of sequences for EmbeddingBag layers.
        # Since we are passing a single sequence (or batch of sequences), 
        # the offsets are just [0, len(seq1), len(seq1)+len(seq2), ...]
        # Here we only pass a single text (batch_size=1), so offset is just [0]
        batch_size = inputs["input_ids"].shape[0]
        offsets_arr = np.zeros((batch_size,), dtype=np.int64)
        ort_inputs["offsets"] = format_input("offsets", offsets_arr)

    # Run inference
    outputs = _ORT_SESSION.run(None, ort_inputs)
    
    # The token embeddings are usually the first output tensor
    # Shape: (batch_size, num_tokens, dim) or (batch_size, dim) for pooled
    embeddings = outputs[0]
    
    # Squeeze batch dimension as we predict for a single text
    embeddings = np.squeeze(embeddings, axis=0)
    
    # If the model returned a pooled 1D embedding (e.g., from EmbeddingBag),
    # expand it to 2D (1, dim) so the script treats it as a single token.
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
        
    return embeddings