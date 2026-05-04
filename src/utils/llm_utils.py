# generative agent utils
import os
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Optional, Any

from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from src.config import llm_temperature  # Import temperature from config
try:
    from peft import PeftConfig, PeftModel
except Exception:
    PeftConfig = None
    PeftModel = None

# Get API keys with better fallback mechanisms
def get_api_key(key_name, file_path=None):
    """Get API key from environment variable or from a file"""
    # First try environment variable
    key = os.getenv(key_name)
    
    # If not found and file_path is provided, try reading from file
    if not key and file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # Extract the key value (format: KEY_NAME=value)
            match = re.search(fr'{key_name}=([^\s"]+)', content)
            if match:
                key = match.group(1).strip()
    
    return key

# Initialize clients with API keys
OPENAI_API_KEY = get_api_key('OPENAI_API_KEY', './OPENAI_API_KEY.env')
if not OPENAI_API_KEY:
    # Try looking for OPENAI_API_KEY.env in current directory
    api_key_files = ['./OPENAI_API_KEY.env', './.env', '../OPENAI_API_KEY.env']
    for file_path in api_key_files:
        if os.path.exists(file_path):
            OPENAI_API_KEY = get_api_key('OPENAI_API_KEY', file_path)
            if OPENAI_API_KEY:
                break

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable or create an OPENAI_API_KEY.env file")

ANTHROPIC_API_KEY = get_api_key('ANTHROPIC_API_KEY')  # Optional

# Optional default path for local LLaMA/Vicuna-style models (not required if you pass a path via --model)
LLAMA_MODEL_PATH = "path/to/llama/model"

# Model alias -> actual API model ID (e.g. HuggingFace, vLLM)
MODEL_ALIASES = {
    "qwen_7B": "Qwen/Qwen2.5-7B-Instruct",
    "gpt-4o-mini": "gpt-4o-mini",
    # Optional convenience aliases for Vicuna models
    "vicuna_7b": "lmsys/vicuna-7b-v1.5",
    "vicuna_13b": "lmsys/vicuna-13b-v1.5",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    # Qwen3 aliases (prefer official IDs when possible)
    "qwen_14b": "nicoboss/Qwen3-14B-Uncensored",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "qwen3_14b_instruct": "Qwen/Qwen3-14B-Instruct",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-14b-instruct": "Qwen/Qwen3-14B-Instruct",
}

oai = OpenAI(api_key=OPENAI_API_KEY.strip())
ant = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
CURRENT_API_SEED: Optional[int] = None


def set_api_seed(seed: Optional[int]) -> None:
    """
    Set a process-wide default API seed used by gen_completion when no
    per-call seed override is provided.
    """
    global CURRENT_API_SEED
    CURRENT_API_SEED = seed

class LLMAdapter:
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = llm_temperature,
        max_tokens: int = 1000,
        seed: Optional[int] = None,
    ) -> str:
        raise NotImplementedError

class OpenAIAdapter(LLMAdapter):
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = llm_temperature,
        max_tokens: int = 1000,
        seed: Optional[int] = None,
    ) -> str:
        try:
            req_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if seed is not None:
                req_kwargs["seed"] = int(seed)

            response = oai.chat.completions.create(
                **req_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            raise e

class ClaudeAdapter(LLMAdapter):
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = llm_temperature,
        max_tokens: int = 1000,
        seed: Optional[int] = None,
    ) -> str:
        try:
            response = ant.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude generation error: {e}")
            raise e

class LlamaAdapter(LLMAdapter):
    def __init__(self, model_path: str):
        # model_path can be a HF repo ID (e.g. "lmsys/vicuna-13b-v1.5") or a local directory
        self.model_path = model_path or LLAMA_MODEL_PATH
        model_path_obj = Path(self.model_path)
        adapter_config_path = model_path_obj / "adapter_config.json"
        # Many modern HF models (e.g. Qwen3) require custom code.
        # trust_remote_code allows Transformers to load the model's repo-provided implementation.
        if adapter_config_path.exists():
            if PeftConfig is None or PeftModel is None:
                raise ValueError(
                    "Detected a PEFT adapter directory, but `peft` is not installed. "
                    "Install with: pip install peft"
                )

            peft_cfg = PeftConfig.from_pretrained(self.model_path)
            base_model_name = peft_cfg.base_model_name_or_path
            if not base_model_name:
                raise ValueError(
                    f"PEFT adapter at {self.model_path} is missing base_model_name_or_path "
                    "in adapter_config.json"
                )

            # Some adapter directories ship incomplete tokenizer artifacts. Prefer base tokenizer.
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # Prefer GPU / mixed precision when available to avoid CPU OOM
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if adapter_config_path.exists():
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            # Older transformers pipelines may reject PeftModel* wrappers.
            # Merge adapter weights into the base model for stable generation.
            self.model = self.model.merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        self.model.eval()
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = llm_temperature,
        max_tokens: int = 1000,
        seed: Optional[int] = None,
    ) -> str:
        try:
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            # Use max_new_tokens to avoid max_length issues; truncate inputs if needed
            response = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=bool(temperature and temperature > 0.0),
                temperature=temperature if temperature and temperature > 0.0 else 1.0,
                truncation=True,
            )
            return response[0]['generated_text']
        except Exception as e:
            print(f"Llama generation error: {e}")
            raise e


# Simple cache so we don't reload large local/HF models on every call
_LLAMA_ADAPTER_CACHE: Dict[str, LlamaAdapter] = {}

def get_llm_client(model_type: str) -> LLMAdapter:
    if model_type.startswith("gpt"):
        return OpenAIAdapter()
    elif model_type.startswith("claude"):
        return ClaudeAdapter()
    else:
        # Fallback: treat anything else as a local/HF causal LM (e.g. Vicuna / LLaMA)
        key = model_type or LLAMA_MODEL_PATH
        if key not in _LLAMA_ADAPTER_CACHE:
            _LLAMA_ADAPTER_CACHE[key] = LlamaAdapter(key)
        return _LLAMA_ADAPTER_CACHE[key]

def gen_completion(messages: List[Dict[str, str]], 
                  model: str = "gpt-4o-mini", 
                  temperature: float = llm_temperature,
                  max_tokens: int = 1000,
                  max_retries: int = 3,
                  retry_delay: float = 2.0,
                  seed: Optional[int] = None) -> str:
    """
    Generate a completion using the specified model.
    
    Args:
        messages: List of message dictionaries
        model: Model to use for generation
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retries on failure
        retry_delay: Delay between retries in seconds
        
    Returns:
        Generated text
    """
    retry_count = 0
    resolved_model = MODEL_ALIASES.get(model, model)
    while retry_count <= max_retries:
        try:
            llm = get_llm_client(resolved_model)
            effective_seed = seed if seed is not None else CURRENT_API_SEED
            return llm.generate(
                messages,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=effective_seed,
            )
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Error generating completion after {max_retries} retries: {e}")
                raise e
            else:
                print(f"API call failed (attempt {retry_count}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay *= 1.5

def simple_gen(prompt: str, model: str = "gpt-4o-mini", temperature: float = llm_temperature) -> str:
    messages = [{"role": "user", "content": prompt}]
    return gen_completion(messages, model, temperature)


def gen_completion_batch(
    messages_list: List[List[Dict[str, str]]],
    model: str = "gpt-4o-mini",
    temperature: float = llm_temperature,
    max_tokens: int = 1000,
    poll_interval: float = 30.0,
    timeout: float = 86400.0,
) -> List[str]:
    """
    Submit messages_list as a single OpenAI Batch API job and block until done.

    Batching sends all prompts in one HTTP call (to the Batch endpoint), so they
    are counted against the Batch API quota instead of the standard RPD limit.
    Returns responses in the same order as messages_list.

    Falls back to serial gen_completion for non-OpenAI models (e.g. local LLMs).
    """
    import tempfile
    import time as _time

    if not messages_list:
        return []

    resolved_model = MODEL_ALIASES.get(model, model)

    # Batch API is only available for OpenAI models; fall back for others
    if not resolved_model.startswith("gpt"):
        print(f"[BatchAPI] {resolved_model} not supported by Batch API; using serial calls")
        return [
            gen_completion(msgs, model=model, temperature=temperature, max_tokens=max_tokens)
            for msgs in messages_list
        ]

    batch_requests = [
        {
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": resolved_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }
        for i, messages in enumerate(messages_list)
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            batch_file = oai.files.create(file=f, purpose="batch")

        batch = oai.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"[BatchAPI] Submitted batch {batch.id} ({len(batch_requests)} requests)")

        start = _time.time()
        while True:
            batch = oai.batches.retrieve(batch.id)
            counts = batch.request_counts
            completed = counts.completed if counts else 0
            total = counts.total if counts else len(batch_requests)
            print(f"[BatchAPI] {batch.id}: {batch.status} ({completed}/{total})")
            if batch.status in {"completed", "failed", "expired", "cancelled"}:
                break
            if _time.time() - start > timeout:
                raise TimeoutError(f"Batch {batch.id} timed out after {timeout}s")
            _time.sleep(poll_interval)

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} ended with status={batch.status}")

        result_text = oai.files.content(batch.output_file_id).text
        results_by_id: Dict[str, str] = {}
        for line in result_text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj["custom_id"]
            if obj.get("error"):
                print(f"[BatchAPI] Error for {cid}: {obj['error']}")
                results_by_id[cid] = ""
            else:
                results_by_id[cid] = (
                    obj["response"]["body"]["choices"][0]["message"]["content"]
                )

        return [results_by_id.get(f"req-{i}", "") for i in range(len(messages_list))]

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

# Prompt utilities
def fill_prompt(prompt: str, placeholders: Dict[str, Any]) -> str:
    for placeholder, value in placeholders.items():
        placeholder_tag = f"!<{placeholder.upper()}>!"
        if placeholder_tag in prompt:
            prompt = prompt.replace(placeholder_tag, str(value))
    return prompt

def make_output_format(modules: List[Dict]) -> str:
    output_format = "Output Format:\n{\n"
    for module in modules:
        if 'name' in module and module['name']:
            output_format += f'    "{module["name"].lower()}": "<your response>",\n'
    output_format = output_format.rstrip(',\n') + "\n}"
    return output_format

def modular_instructions(modules: List[Dict]) -> str:
    prompt = ""
    step_count = 0
    for module in modules:
        if 'name' in module:
            step_count += 1
            prompt += f"Step {step_count} ({module['name']}): {module['instruction']}\n"
        else:
            prompt += f"{module['instruction']}\n"
    prompt += "\n"
    prompt += make_output_format(modules)
    return prompt

def parse_json(response: str, target_keys: Optional[List[str]] = None) -> Dict:
    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    cleaned_response = response[json_start:json_end].replace('\\"', '"')
    
    try:
        parsed = json.loads(cleaned_response)
        if target_keys:
            parsed = {key: parsed.get(key, "") for key in target_keys}
        return parsed
    except json.JSONDecodeError:
        print("JSON parsing failed. Using regex fallback.")
        # print(f"Response: {cleaned_response}")
        parsed = {}
        for key_match in re.finditer(r'"(\w+)":\s*', cleaned_response):
            key = key_match.group(1)
            if target_keys and key not in target_keys:
                continue
            value_start = key_match.end()
            if cleaned_response[value_start] == '"':
                value_match = re.search(r'"(.*?)"(?:,|\s*})', 
                                      cleaned_response[value_start:])
                if value_match:
                    parsed[key] = value_match.group(1)
            elif cleaned_response[value_start] == '{':
                nested_json = re.search(r'(\{.*?\})(?:,|\s*})', 
                                      cleaned_response[value_start:], re.DOTALL)
                if nested_json:
                    try:
                        parsed[key] = json.loads(nested_json.group(1))
                    except json.JSONDecodeError:
                        parsed[key] = {}
            else:
                value_match = re.search(r'([^,}]+)(?:,|\s*})', 
                                      cleaned_response[value_start:])
                if value_match:
                    parsed[key] = value_match.group(1).strip()
        
        if target_keys:
            parsed = {key: parsed.get(key, "") for key in target_keys}
        return parsed

def mod_gen(modules: List[Dict], placeholders: Dict, 
            target_keys: Optional[List[str]] = None,
            model: str = "gpt-4o-mini") -> Dict:
    prompt = modular_instructions(modules)
    filled = fill_prompt(prompt, placeholders)
    response = simple_gen(filled, model)
    if len(response) == 0:
        print("Error: response was empty")
        return {}
    if target_keys == None:
        target_keys = [module["name"].lower() for module in modules if "name" in module]
    parsed = parse_json(response, target_keys)
    return parsed 