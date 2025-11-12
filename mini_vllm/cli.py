"""CLI entry point for mini-vLLM."""

import argparse
from pathlib import Path

import uvicorn
import yaml
from loguru import logger

from mini_vllm.engine.loader import ModelLoader
from mini_vllm.engine.runtime import InferenceRuntime
from mini_vllm.server.http import create_app
from mini_vllm.utils.logging import setup_logging


def load_config(config_path: Path) -> dict:
    """Load YAML config file.

    Args:
        config_path: Path to config file

    Returns:
        Config dict
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def serve(args):
    """Start the server."""
    # Load configs
    config_dir = Path(args.config_dir)
    server_config_path = config_dir / "server.yaml"
    # Determine model config path
    if args.model:
        model_config_path = config_dir / f"model-{args.model}.yaml"
    else:
        # Default priority: tinyllama > llama > gpt2
        model_config_path = config_dir / "model-tinyllama.yaml"
        if not model_config_path.exists():
            model_config_path = config_dir / "model-llama.yaml"
        if not model_config_path.exists():
            model_config_path = config_dir / "model-gpt2.yaml"

    if not server_config_path.exists():
        raise FileNotFoundError(f"Server config not found: {server_config_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    server_config = load_config(server_config_path)
    model_config = load_config(model_config_path)

    # Setup logging
    log_config = server_config.get("logging", {})
    # Use CLI argument if provided, otherwise use config file, otherwise default to INFO
    log_level = args.log_level or log_config.get("level", "INFO")
    setup_logging(
        level=log_level,
        format_string=log_config.get("format"),
    )

    logger.info("Starting mini-vLLM server...")
    logger.info(f"Server config: {server_config_path}")
    logger.info(f"Model config: {model_config_path}")

    # Load model
    model_cfg = model_config["model"]
    quantization_config = model_cfg.get("quantization", {})
    loader = ModelLoader(
        model_name=model_cfg["name"],
        checkpoint_path=model_cfg.get("checkpoint_path"),
        dtype=model_cfg.get("dtype", "float32"),
        quantization_config=quantization_config,
    )
    loader.load()

    # Create runtime
    # Disable KV cache on macOS CPU due to bus error bug
    # MPS (Apple Silicon) can use cache safely
    import platform

    use_cache = True
    if platform.system() == "Darwin" and loader.device == "cpu":
        use_cache = False
        logger.warning("KV cache disabled due to macOS CPU compatibility issue")
    elif loader.device == "mps":
        logger.info("Using MPS backend - KV cache enabled")

    runtime = InferenceRuntime(
        model=loader.get_model(),
        tokenizer=loader.get_tokenizer(),
        device=loader.device,
        use_cache=use_cache,
    )

    # Create app
    app = create_app(runtime=runtime)

    # Start server
    server_cfg = server_config["server"]
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)

    logger.info(f"Server starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="mini-vLLM inference server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the inference server")
    serve_parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing config files (default: configs)",
    )
    serve_parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["gpt2", "llama", "tinyllama"],
        help="Model to use: gpt2, llama, or tinyllama (default: auto-detect)",
    )
    serve_parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (overrides config file): DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    args = parser.parse_args()

    if args.command == "serve":
        serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
