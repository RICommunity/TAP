import os
import time
import json
from typing import Dict, List, Optional

import urllib3
from dotenv import load_dotenv

import common

# Load OPENROUTER_API_KEY (and any other keys) from the local .env file.
load_dotenv()

# ---------------------------------------------------------------------------
# Imports for the non-OpenRouter back-ends are commented out: those models
# (local HuggingFace checkpoints, Google PaLM/Gemini, self-hosted endpoints,
# the direct OpenAI API) are not used in this OpenRouter-only setup.
# ---------------------------------------------------------------------------
# import openai
# import anthropic
# import torch
# import gc
# import google.generativeai as genai
# from copy import deepcopy
# from config import LLAMA_API_LINK, VICUNA_API_LINK


class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenRouter back-end.
#
# OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible chat
# completions API, so a single class can serve as attacker, target and
# evaluator. Friendly CLI names are mapped to OpenRouter model ids below;
# any other string is passed through to OpenRouter verbatim, so you can also
# use a raw id such as "openai/gpt-4o-mini" directly.
# ---------------------------------------------------------------------------

# Browse the full catalogue at https://openrouter.ai/models
# NOTE: ids are "provider/model" (no leading "openrouter/"). OpenRouter retires
# models over time, so verify with `GET https://openrouter.ai/api/v1/models`.
# These aliases were verified against the live catalogue on 2026-06-23.
OPENROUTER_MODELS = {
    # OpenAI
    "gpt-4o-mini":       "openai/gpt-4o-mini",
    "gpt-4.1":           "openai/gpt-4.1",
    "gpt-5-mini":        "openai/gpt-5-mini",
    "gpt-4o":            "openai/gpt-4o",
    "gpt-4-turbo":       "openai/gpt-4-turbo",
    "gpt-3.5-turbo":     "openai/gpt-3.5-turbo",
    # Anthropic
    "claude-opus-4.8":   "anthropic/claude-opus-4.8",
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "claude-haiku-4.5":  "anthropic/claude-haiku-4.5",
    "claude-3-haiku":    "anthropic/claude-3-haiku",
    # Google
    "gemini-2.5-pro":    "google/gemini-2.5-pro",
    "gemini-2.5-flash":  "google/gemini-2.5-flash",
    # Meta Llama
    "llama-3.3-70b":     "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-70b":     "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b":      "meta-llama/llama-3.1-8b-instruct",
    # Mistral
    "mixtral-8x22b":     "mistralai/mixtral-8x22b-instruct",
    # Microsoft
    "wizardlm-2-8x22b":  "microsoft/wizardlm-2-8x22b",
}


def is_openrouter_model(model_name: str) -> bool:
    """Every real model in this setup runs through OpenRouter's
    OpenAI-compatible chat API, so all of them use the OpenAI message format.
    Only the sentinel "no-evaluator" is not an OpenRouter model."""
    return model_name != "no-evaluator"


class OpenRouter(LanguageModel):
    API_RETRY_SLEEP = 3
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 100

    API_HOST_LINK = "https://openrouter.ai/api/v1/chat/completions"

    @staticmethod
    def _should_retry_error(error) -> bool:
        """Only retry transient OpenRouter failures (rate limits, overload)."""
        if not isinstance(error, dict):
            return False
        code = error.get("code")
        if code in (429, 500, 502, 503, 504):
            return True
        message = str(error.get("message", "")).lower()
        return any(token in message for token in ("rate limit", "timeout", "overloaded", "temporarily"))

    def __init__(self, model_name):
        super().__init__(model_name)
        # Resolve a friendly alias to its OpenRouter id, otherwise pass through.
        api_model_id = OPENROUTER_MODELS.get(model_name, model_name)
        # OpenRouter ids are "provider/model"; strip a stray "openrouter/"
        # prefix so e.g. "openrouter/microsoft/wizardlm-2-8x22b" still works.
        if api_model_id.startswith("openrouter/"):
            api_model_id = api_model_id[len("openrouter/"):]
        self.api_model_id = api_model_id
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to stupid-TAP/.env "
                "(see .env.example)."
            )
        self._http = urllib3.PoolManager()

    def generate(self, conv: List[Dict],
                max_n_tokens: int,
                temperature: float,
                top_p: float,
                role: str = "model",
                batch_index: Optional[int] = None):
        '''
        Args:
            conv: List of dictionaries, OpenAI API chat-message format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT

        body = {
            "model": self.api_model_id,
            "messages": conv,
            "max_tokens": max_n_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        for attempt in range(self.API_MAX_RETRY):
            try:
                resp = self._http.request(
                    "POST",
                    self.API_HOST_LINK,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        # Optional attribution headers recommended by OpenRouter.
                        "HTTP-Referer": "https://github.com/RICommunity/TAP",
                        "X-Title": "TAP",
                    },
                    body=json.dumps(body).encode("utf-8"),
                    timeout=urllib3.Timeout(self.API_TIMEOUT),
                )

                if resp.status >= 400:
                    print(f"OpenRouter HTTP {resp.status} for {self.api_model_id}", flush=True)
                    if attempt + 1 < self.API_MAX_RETRY:
                        time.sleep(self.API_RETRY_SLEEP)
                        continue
                    break

                resp_json = json.loads(resp.data.decode("utf-8"))

                if "error" in resp_json:
                    err = resp_json["error"]
                    print("OpenRouter error:", err, flush=True)
                    if self._should_retry_error(err) and attempt + 1 < self.API_MAX_RETRY:
                        time.sleep(self.API_RETRY_SLEEP)
                        continue
                    break

                choice = resp_json["choices"][0]
                content = choice.get("message", {}).get("content")
                finish_reason = choice.get("finish_reason")

                if content is None or not str(content).strip():
                    # Content filters / refusals return empty bodies — retrying won't help.
                    print(
                        f"OpenRouter empty content for {self.api_model_id} "
                        f"(finish_reason={finish_reason!r})",
                        flush=True,
                    )
                    break

                output = content
                break
            except Exception as e:
                print('exception!', type(e), e, flush=True)
                if attempt + 1 < self.API_MAX_RETRY:
                    time.sleep(self.API_RETRY_SLEEP)
                    continue
                break

            time.sleep(self.API_QUERY_SLEEP)
        common.debug_model_io(
            role,
            self.api_model_id,
            conv,
            output,
            max_n_tokens=max_n_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_index=batch_index,
        )
        return output

    def batched_generate(self,
                        convs_list: List[List[Dict]],
                        max_n_tokens: int,
                        temperature: float,
                        top_p: float = 1.0,
                        role: str = "model"):
        return [
            self.generate(
                conv,
                max_n_tokens,
                temperature,
                top_p,
                role=role,
                batch_index=i if len(convs_list) > 1 else None,
            )
            for i, conv in enumerate(convs_list)
        ]


class OpenRouterEmbedder:
    """Text embedding client for OpenRouter's OpenAI-compatible /embeddings API.

    Used by the Phase-2 memory layer to embed attack goals (read path) and
    successful adversarial prompts (write path). Vectors are produced here and
    stored directly in Chroma, so the vector DB never needs a local embedding
    model (no torch / onnxruntime dependency).
    """

    API_HOST_LINK = "https://openrouter.ai/api/v1/embeddings"
    API_RETRY_SLEEP = 3
    API_MAX_RETRY = 5
    API_TIMEOUT = 100
    # OpenRouter caps embedding batches; stay well under it.
    MAX_BATCH = 96

    def __init__(self, model_name: str = "openai/text-embedding-3-small"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to stupid-TAP/.env "
                "(see .env.example)."
            )
        self._http = urllib3.PoolManager()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input string (order preserved)."""
        if not texts:
            return []

        vectors: List[List[float]] = []
        for left in range(0, len(texts), self.MAX_BATCH):
            batch = texts[left:left + self.MAX_BATCH]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        body = {
            "model": self.model_name,
            "input": batch,
            "encoding_format": "float",
        }
        last_error = None
        for attempt in range(self.API_MAX_RETRY):
            try:
                resp = self._http.request(
                    "POST",
                    self.API_HOST_LINK,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/RICommunity/TAP",
                        "X-Title": "TAP",
                    },
                    body=json.dumps(body).encode("utf-8"),
                    timeout=urllib3.Timeout(self.API_TIMEOUT),
                )

                if resp.status >= 400:
                    last_error = f"HTTP {resp.status}: {resp.data.decode('utf-8')[:300]}"
                    if attempt + 1 < self.API_MAX_RETRY:
                        time.sleep(self.API_RETRY_SLEEP)
                        continue
                    break

                resp_json = json.loads(resp.data.decode("utf-8"))
                if "error" in resp_json:
                    last_error = str(resp_json["error"])
                    if attempt + 1 < self.API_MAX_RETRY:
                        time.sleep(self.API_RETRY_SLEEP)
                        continue
                    break

                # Response order matches input order per the OpenAI schema, but
                # sort by "index" defensively in case a provider reorders them.
                data = sorted(resp_json["data"], key=lambda d: d.get("index", 0))
                return [d["embedding"] for d in data]
            except Exception as e:  # noqa: BLE001 - surface after retries
                last_error = f"{type(e).__name__}: {e}"
                if attempt + 1 < self.API_MAX_RETRY:
                    time.sleep(self.API_RETRY_SLEEP)
                    continue
                break

        raise RuntimeError(
            f"OpenRouter embeddings failed after {self.API_MAX_RETRY} attempts. "
            f"Last error: {last_error}"
        )


# ===========================================================================
# The classes below target back-ends that are NOT available through OpenRouter
# (local GPU inference, self-hosted endpoints, Google's native SDK, and the
# direct OpenAI API). They are commented out for this OpenRouter-only setup;
# re-enable them (and the matching imports above and in requirements.txt) if
# you bring those back-ends online.
# ===========================================================================
#
# class HuggingFace(LanguageModel):
#     def __init__(self,model_name, model, tokenizer):
#         self.model_name = model_name
#         self.model = model
#         self.tokenizer = tokenizer
#         self.eos_token_ids = [self.tokenizer.eos_token_id]
#
#     def batched_generate(self,
#                         full_prompts_list,
#                         max_n_tokens: int,
#                         temperature: float,
#                         top_p: float = 1.0,):
#         inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
#         inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
#
#         # Batch generation
#         if temperature > 0:
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_n_tokens,
#                 do_sample=True,
#                 temperature=temperature,
#                 eos_token_id=self.eos_token_ids,
#                 top_p=top_p,
#             )
#         else:
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_n_tokens,
#                 do_sample=False,
#                 eos_token_id=self.eos_token_ids,
#                 top_p=1,
#                 temperature=1, # To prevent warning messages
#             )
#
#         # If the model is not an encoder-decoder type, slice off the input tokens
#         if not self.model.config.is_encoder_decoder:
#             output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
#
#         # Batch decoding
#         outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#
#         for key in inputs:
#             inputs[key].to('cpu')
#         output_ids.to('cpu')
#         del inputs, output_ids
#         gc.collect()
#         torch.cuda.empty_cache()
#
#         return outputs_list
#
#     def extend_eos_tokens(self):
#         # Add closing braces for Vicuna/Llama eos when using attacker model
#         self.eos_token_ids.extend([
#             self.tokenizer.encode("}")[1],
#             29913,
#             9092,
#             16675])
#
#
# class APIModel(LanguageModel):
#
#     API_HOST_LINK = "ADD_LINK"
#     API_RETRY_SLEEP = 10
#     API_ERROR_OUTPUT = "$ERROR$"
#     API_QUERY_SLEEP = 0.5
#     API_MAX_RETRY = 20
#
#     API_TIMEOUT = 100
#
#     MODEL_API_KEY = os.getenv("MODEL_API_KEY")
#
#     API_HOST_LINK = ''
#
#     def generate(self, conv: List[Dict],
#                 max_n_tokens: int,
#                 temperature: float,
#                 top_p: float):
#         '''
#         Args:
#             conv: List of dictionaries, OpenAI API format
#             max_n_tokens: int, max number of tokens to generate
#             temperature: float, temperature for sampling
#             top_p: float, top p for sampling
#         Returns:
#             str: generated response
#         '''
#         output = self.API_ERROR_OUTPUT
#
#         for _ in range(self.API_MAX_RETRY):
#             try:
#
#                 # Batch generation
#                 if temperature > 0:
#                     # Attack model
#                     json = {
#                         "top_p": top_p,
#                         "num_beams": 1,
#                         "temperature": temperature,
#                         "do_sample": True,
#                         "prompt": '',
#                         "max_new_tokens": max_n_tokens,
#                         "system_prompt": conv,
#                     }
#                 else:
#                     # Target model
#                     json = {
#                         "top_p": 1,
#                         "num_beams": 1,
#                         "temperature": 1, # To prevent warning messages
#                         "do_sample": False,
#                         "prompt": '',
#                         "max_new_tokens": max_n_tokens,
#                         "system_prompt": conv,
#                     }
#
#                     # Do not use extra end-of-string tokens in target mode
#                     if 'llama' in self.model_name:
#                         json['extra_eos_tokens'] = 0
#
#
#                 if 'llama' in self.model_name:
#                     # No system prompt for the Llama model
#                     assert json['prompt'] == ''
#                     json['prompt'] = deepcopy(json['system_prompt'])
#                     del json['system_prompt']
#
#                 resp = urllib3.request(
#                             "POST",
#                             self.API_HOST_LINK,
#                             headers={"Authorization": f"Api-Key {self.MODEL_API_KEY}"},
#                             timeout=urllib3.Timeout(self.API_TIMEOUT),
#                             json=json,
#                 )
#
#                 resp_json = resp.json()
#
#                 if 'vicuna' in self.model_name:
#                     if 'error' in resp_json:
#                         print(self.API_ERROR_OUTPUT)
#
#                     output = resp_json['output']
#
#                 else:
#                     output = resp_json
#
#                 if type(output) == type([]):
#                     output = output[0]
#
#                 break
#             except Exception as e:
#                 print('exception!', type(e), e)
#                 time.sleep(self.API_RETRY_SLEEP)
#
#             time.sleep(self.API_QUERY_SLEEP)
#         return output
#
#     def batched_generate(self,
#                         convs_list: List[List[Dict]],
#                         max_n_tokens: int,
#                         temperature: float,
#                         top_p: float = 1.0,):
#         return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
#
# class APIModelLlama7B(APIModel):
#     API_HOST_LINK = LLAMA_API_LINK
#     MODEL_API_KEY = os.getenv("LLAMA_API_KEY")
#
# class APIModelVicuna13B(APIModel):
#     API_HOST_LINK = VICUNA_API_LINK
#     MODEL_API_KEY = os.getenv("VICUNA_API_KEY")
#
# class GPT(LanguageModel):
#     API_RETRY_SLEEP = 10
#     API_ERROR_OUTPUT = "$ERROR$"
#     API_QUERY_SLEEP = 0.5
#     API_MAX_RETRY = 20
#     API_TIMEOUT = 20
#
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#
#     def generate(self, conv: List[Dict],
#                 max_n_tokens: int,
#                 temperature: float,
#                 top_p: float):
#         '''
#         Args:
#             conv: List of dictionaries, OpenAI API format
#             max_n_tokens: int, max number of tokens to generate
#             temperature: float, temperature for sampling
#             top_p: float, top p for sampling
#         Returns:
#             str: generated response
#         '''
#         output = self.API_ERROR_OUTPUT
#         for _ in range(self.API_MAX_RETRY):
#             try:
#
#                 response = openai.ChatCompletion.create(
#                             model = self.model_name,
#                             messages = conv,
#                             max_tokens = max_n_tokens,
#                             temperature = temperature,
#                             top_p = top_p,
#                             request_timeout = self.API_TIMEOUT,
#                             )
#                 output = response["choices"][0]["message"]["content"]
#                 break
#             except Exception as e:
#                 print(type(e), e)
#                 time.sleep(self.API_RETRY_SLEEP)
#
#             time.sleep(self.API_QUERY_SLEEP)
#         return output
#
#     def batched_generate(self,
#                         convs_list: List[List[Dict]],
#                         max_n_tokens: int,
#                         temperature: float,
#                         top_p: float = 1.0,):
#         return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
#
# class PaLM():
#     API_RETRY_SLEEP = 10
#     API_ERROR_OUTPUT = "$ERROR$"
#     API_QUERY_SLEEP = 1
#     API_MAX_RETRY = 5
#     API_TIMEOUT = 20
#     default_output = "I'm sorry, but I cannot assist with that request."
#     API_KEY = os.getenv("PALM_API_KEY")
#
#     def __init__(self, model_name) -> None:
#         self.model_name = model_name
#         genai.configure(api_key=self.API_KEY)
#
#     def generate(self, conv: List,
#                 max_n_tokens: int,
#                 temperature: float,
#                 top_p: float):
#         '''
#         Args:
#             conv: List of dictionaries,
#             max_n_tokens: int, max number of tokens to generate
#             temperature: float, temperature for sampling
#             top_p: float, top p for sampling
#         Returns:
#             str: generated response
#         '''
#         output = self.API_ERROR_OUTPUT
#         for _ in range(self.API_MAX_RETRY):
#             try:
#                 completion = genai.chat(
#                     messages=conv,
#                     temperature=temperature,
#                     top_p=top_p
#                 )
#                 output = completion.last
#
#                 if output is None:
#                     # If PaLM refuses to output and returns None, we replace it with a default output
#                     output = self.default_output
#                 else:
#                     # Use this approximation since PaLM does not allow
#                     # to specify max_tokens. Each token is approximately 4 characters.
#                     output = output[:(max_n_tokens*4)]
#                 break
#             except Exception as e:
#                 print(type(e), e)
#                 time.sleep(self.API_RETRY_SLEEP)
#
#             time.sleep(self.API_QUERY_SLEEP)
#         return output
#
#     def batched_generate(self,
#                         convs_list: List[List[Dict]],
#                         max_n_tokens: int,
#                         temperature: float,
#                         top_p: float = 1.0,):
#         return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
#
#
# class GeminiPro():
#     API_RETRY_SLEEP = 10
#     API_ERROR_OUTPUT = "$ERROR$"
#     API_QUERY_SLEEP = 1
#     API_MAX_RETRY = 5
#     API_TIMEOUT = 20
#     default_output = "I'm sorry, but I cannot assist with that request."
#     API_KEY = os.getenv("PALM_API_KEY")
#
#     def __init__(self, model_name) -> None:
#         self.model_name = model_name
#         genai.configure(api_key=self.API_KEY)
#
#     def generate(self, conv: List,
#                 max_n_tokens: int,
#                 temperature: float,
#                 top_p: float):
#         '''
#         Args:
#             conv: List of dictionaries,
#             max_n_tokens: int, max number of tokens to generate
#             temperature: float, temperature for sampling
#             top_p: float, top p for sampling
#         Returns:
#             str: generated response
#         '''
#         output = self.API_ERROR_OUTPUT
#         for _ in range(self.API_MAX_RETRY):
#             try:
#                 model = genai.GenerativeModel(self.model_name)
#                 output = model.generate_content(
#                     contents = conv,
#                     generation_config = genai.GenerationConfig(
#                         candidate_count = 1,
#                         temperature = temperature,
#                         top_p = top_p,
#                         max_output_tokens=max_n_tokens,
#                     )
#                 )
#
#                 if output is None:
#                     # If PaLM refuses to output and returns None, we replace it with a default output
#                     output = self.default_output
#                 else:
#                     # Use this approximation since PaLM does not allow
#                     # to specify max_tokens. Each token is approximately 4 characters.
#                     output = output.text
#                 break
#             except Exception as e:
#                 print(type(e), e)
#                 time.sleep(self.API_RETRY_SLEEP)
#
#             time.sleep(self.API_QUERY_SLEEP)
#         return output
#
#     def batched_generate(self,
#                         convs_list: List[List[Dict]],
#                         max_n_tokens: int,
#                         temperature: float,
#                         top_p: float = 1.0,):
#         return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
