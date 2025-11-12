"""Single-request inference runtime (prefill + decode loop)."""

from typing import List

import torch
from loguru import logger


class InferenceRuntime:
    """Simple runtime for single-request inference."""

    def __init__(self, model, tokenizer, device: str = "cpu", use_cache: bool = True):
        """Initialize runtime.

        Args:
            model: Loaded model (from loader.py)
            tokenizer: Loaded tokenizer (from loader.py)
            device: Device to run on
            use_cache: Whether to use KV cache (disable for macOS CPU workaround)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_cache = use_cache

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_sequences: List[str] | None = None,
    ) -> tuple[str, List[str]]:
        """Generate text from prompt (single request, no batching).

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = disabled)
            top_p: Nucleus sampling (None = disabled)
            stop_sequences: Stop sequences (None = disabled)

        Returns:
            Tuple of (generated_text, list_of_tokens)
        """
        logger.debug(
            f"[generate] Starting generation: prompt_len={len(prompt)}, max_new_tokens={max_new_tokens}, device={self.device}"
        )

        # Apply chat template if available (for instruction-tuned models like Llama)
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            logger.debug("[generate] Applying chat template")
            # Format as a single user message
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            logger.debug(f"[generate] Formatted prompt length: {len(formatted_prompt)}")
        else:
            formatted_prompt = prompt

        # Tokenize input
        logger.debug("[generate] Tokenizing input...")
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        logger.debug(f"[generate] Tokenizer output keys: {inputs.keys()}")

        input_ids = inputs["input_ids"].to(self.device)
        logger.debug(
            f"[generate] input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}, device: {input_ids.device}"
        )

        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            logger.debug(
                f"[generate] attention_mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}"
            )
        else:
            logger.debug("[generate] No attention_mask provided")

        logger.debug(f"[generate] Input tokens count: {input_ids.shape[1]}")

        # Prefill: forward pass on input tokens
        logger.debug("[generate] Starting prefill forward pass...")
        try:
            with torch.no_grad():
                logger.debug(
                    f"[generate] Calling model.forward with input_ids shape: {input_ids.shape}"
                )
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=self.use_cache,
                )
                logger.debug("[generate] Model forward pass completed")
                logger.debug(
                    f"[generate] Outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}"
                )
                logger.debug(
                    f"[generate] Outputs logits shape: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}"
                )

                if self.use_cache:
                    past_key_values = outputs.past_key_values
                    logger.debug(
                        f"[generate] past_key_values type: {type(past_key_values)}, length: {len(past_key_values) if past_key_values else 'None'}"
                    )
                else:
                    past_key_values = None
                    logger.debug(
                        "[generate] use_cache disabled, not storing past_key_values"
                    )

                next_token_logits = outputs.logits[:, -1, :]  # Last token logits
                logger.debug(
                    f"[generate] next_token_logits shape: {next_token_logits.shape}, dtype: {next_token_logits.dtype}, device: {next_token_logits.device}"
                )
        except Exception as e:
            logger.debug(
                f"[generate] ERROR in prefill forward pass: {e}", exc_info=True
            )
            raise

        # Initialize generation
        logger.debug("[generate] Initializing generation state...")
        generated_tokens = []
        generated_text = ""
        current_ids = input_ids.clone()
        logger.debug(f"[generate] current_ids shape after clone: {current_ids.shape}")

        # Decode loop: generate one token at a time
        logger.debug(f"[generate] Starting decode loop (max {max_new_tokens} steps)...")
        for step in range(max_new_tokens):
            logger.debug(f"[generate] === Decode step {step}/{max_new_tokens} ===")

            # Sample next token
            logger.debug(
                f"[generate] Sampling token from logits shape: {next_token_logits.shape}"
            )
            try:
                next_token_id = self._sample_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                logger.debug(f"[generate] Sampled token_id: {next_token_id}")
            except Exception as e:
                logger.debug(f"[generate] ERROR in _sample_token: {e}", exc_info=True)
                raise

            # Decode token
            logger.debug(f"[generate] Decoding token_id {next_token_id} to text...")
            try:
                token = self.tokenizer.decode(
                    [next_token_id], skip_special_tokens=False
                )
                logger.debug(f"[generate] Decoded token: {repr(token)}")
            except Exception as e:
                logger.debug(
                    f"[generate] ERROR in tokenizer.decode: {e}", exc_info=True
                )
                raise

            generated_tokens.append(token)
            generated_text += token
            logger.debug(
                f"[generate] Generated text so far: {repr(generated_text[:100])}"
            )

            # Check stop conditions
            if self._check_stop(generated_text, stop_sequences):
                logger.debug(f"[generate] Stopped at step {step} due to stop sequence")
                break

            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                logger.debug(f"[generate] Stopped at step {step} due to EOS token")
                break

            # Prepare for next iteration
            logger.debug("[generate] Preparing next iteration...")
            try:
                next_token_ids = torch.tensor([[next_token_id]], device=self.device)
                logger.debug(
                    f"[generate] next_token_ids shape: {next_token_ids.shape}, dtype: {next_token_ids.dtype}, device: {next_token_ids.device}"
                )

                current_ids = torch.cat([current_ids, next_token_ids], dim=1)
                logger.debug(
                    f"[generate] current_ids shape after concat: {current_ids.shape}"
                )
            except Exception as e:
                logger.debug(
                    f"[generate] ERROR preparing next_token_ids: {e}", exc_info=True
                )
                raise

            # Decode step: forward pass with past_key_values
            logger.debug("[generate] Starting decode forward pass...")
            try:
                # Log detailed info about past_key_values before the call
                if past_key_values is not None:
                    logger.debug(
                        f"[generate] past_key_values type: {type(past_key_values)}"
                    )
                    if hasattr(past_key_values, "__len__"):
                        logger.debug(
                            f"[generate] past_key_values length: {len(past_key_values)}"
                        )
                    if hasattr(past_key_values, "key_cache"):
                        logger.debug(
                            f"[generate] past_key_values has key_cache: {hasattr(past_key_values, 'key_cache')}"
                        )

                with torch.no_grad():
                    logger.debug(
                        f"[generate] Calling model.forward with input_ids shape: {next_token_ids.shape}, past_key_values present: {past_key_values is not None}"
                    )
                    logger.debug(
                        f"[generate] input_ids device: {next_token_ids.device}, dtype: {next_token_ids.dtype}"
                    )

                    # Workaround for macOS CPU: ensure tensors are contiguous
                    next_token_ids = next_token_ids.contiguous()
                    logger.debug("[generate] Made next_token_ids contiguous")

                    # Workaround for macOS CPU bus error: if use_cache is disabled,
                    # re-run forward pass on full sequence instead of using past_key_values
                    if not self.use_cache:
                        logger.debug(
                            "[generate] use_cache disabled, running full forward pass"
                        )
                        # Re-run on full sequence
                        outputs = self.model(
                            input_ids=current_ids,
                            attention_mask=None,
                            use_cache=False,
                        )
                    else:
                        outputs = self.model(
                            input_ids=next_token_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    logger.debug("[generate] Decode forward pass completed")
                    logger.debug(
                        f"[generate] Outputs logits shape: {outputs.logits.shape}"
                    )

                    if self.use_cache:
                        past_key_values = outputs.past_key_values
                        logger.debug(
                            f"[generate] Updated past_key_values, length: {len(past_key_values) if past_key_values else 'None'}"
                        )
                    else:
                        past_key_values = None

                    next_token_logits = outputs.logits[:, -1, :]
                    logger.debug(
                        f"[generate] next_token_logits shape after decode: {next_token_logits.shape}, dtype: {next_token_logits.dtype}"
                    )
            except Exception as e:
                logger.debug(
                    f"[generate] ERROR in decode forward pass at step {step}: {e}",
                    exc_info=True,
                )
                raise

        logger.debug(
            f"[generate] Generation complete: Generated {len(generated_tokens)} tokens"
        )
        return generated_text, generated_tokens

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> int:
        """Sample a token from logits using greedy decoding.

        Args:
            logits: Shape [vocab_size]
            temperature: Temperature for sampling
            top_k: Top-k filtering (not yet implemented)
            top_p: Nucleus sampling (not yet implemented)

        Returns:
            Token ID
        """
        logger.debug(
            f"[_sample_token] Input logits shape: {logits.shape}, dtype: {logits.dtype}, device: {logits.device}"
        )
        logger.debug(
            f"[_sample_token] logits dim: {logits.dim()}, numel: {logits.numel()}"
        )

        # Apply temperature
        if temperature != 1.0:
            logger.debug(f"[_sample_token] Applying temperature: {temperature}")
            logits = logits / temperature
            logger.debug(
                f"[_sample_token] Logits after temperature: shape={logits.shape}, min={logits.min().item():.4f}, max={logits.max().item():.4f}"
            )

        # Simple greedy decoding (argmax)
        if top_k is not None or top_p is not None:
            logger.warning("top_k and top_p not yet implemented, using greedy")

        logger.debug("[_sample_token] Computing argmax...")
        try:
            argmax_result = logits.argmax(dim=-1)
            logger.debug(
                f"[_sample_token] argmax result shape: {argmax_result.shape}, dtype: {argmax_result.dtype}"
            )

            token_id = argmax_result.item()
            logger.debug(f"[_sample_token] token_id: {token_id}")
        except Exception as e:
            logger.debug(f"[_sample_token] ERROR in argmax/item: {e}", exc_info=True)
            raise

        return token_id

    def _check_stop(self, text: str, stop_sequences: List[str] | None) -> bool:
        """Check if any stop sequence appears in generated text."""
        if stop_sequences is None:
            return False
        for stop_seq in stop_sequences:
            if stop_seq in text:
                return True
        return False
