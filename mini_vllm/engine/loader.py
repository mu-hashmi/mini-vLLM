"""Load model and tokenizer from HuggingFace."""

import platform
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import bitsandbytes for quantization
try:
    from transformers import BitsAndBytesConfig

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


class ModelLoader:
    """Loads and manages a HuggingFace model and tokenizer."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        dtype: str = "float32",
        device: str | None = None,
        quantization_config: Optional[dict] = None,
    ):
        """Initialize model loader.

        Args:
            model_name: HuggingFace model name (e.g., "gpt2")
            checkpoint_path: Optional path to local checkpoint. If None, uses model_name.
            dtype: Model dtype ("float32", "float16", "bfloat16")
            device: Device to load on ("cpu", "cuda", etc.). If None, auto-detects.
            quantization_config: Optional dict with quantization settings.
                Example: {"enabled": True, "bits": 4}
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path or model_name
        self.dtype = dtype
        self.quantization_config = quantization_config or {}
        # Auto-detect device: prefer CUDA > MPS > CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                # MPS requires macOS 14.0+ (Sonoma)
                # is_available() checks this automatically
                self.device = "mps"
                logger.debug("MPS backend available")
            else:
                # Check if MPS is built but not available (likely macOS < 14.0)
                if torch.backends.mps.is_built() and platform.system() == "Darwin":
                    mac_version = platform.mac_ver()[0]
                    logger.info(
                        f"MPS backend is built but requires macOS 14.0+. "
                        f"Current version: {mac_version}. Falling back to CPU."
                    )
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.checkpoint_path} on {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine torch dtype
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.dtype, torch.float32)

        # Setup quantization if requested
        quantization_config_obj = None
        if self.quantization_config.get("enabled", False):
            if not BITSANDBYTES_AVAILABLE:
                logger.warning(
                    "bitsandbytes not available. Install with: pip install bitsandbytes"
                )
            elif self.device == "cpu":
                logger.warning(
                    "Quantization requires CUDA. Falling back to regular dtype."
                )
            else:
                bits = self.quantization_config.get("bits", 4)
                if bits == 4:
                    quantization_config_obj = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info("Using 4-bit quantization")
                elif bits == 8:
                    quantization_config_obj = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    logger.info("Using 8-bit quantization")
                else:
                    logger.warning(f"Unsupported quantization bits: {bits}")

        # Load model
        model_kwargs = {
            "dtype": torch_dtype if quantization_config_obj is None else None,
        }
        if quantization_config_obj is not None:
            model_kwargs["quantization_config"] = quantization_config_obj
            # For quantized models, device_map handles device placement
            model_kwargs["device_map"] = "auto"
        else:
            # For non-quantized, we'll move to device manually
            model_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            **model_kwargs,
        )

        # Move to device if not using quantization
        if quantization_config_obj is None:
            self.model.to(self.device)

        self.model.eval()

        logger.info(f"Model loaded: {self.model.config.name_or_path}")
        logger.info(f"Model dtype: {torch_dtype}, device: {self.device}")

    def get_model(self):
        """Get the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model

    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self.tokenizer
