"""Synthetic load generator: spawn N clients and record CSV."""

import argparse
import asyncio
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

try:
    import httpx
    import websockets
except ImportError:
    print(
        "Error: httpx and websockets are required. Install with: pip install httpx websockets"
    )
    sys.exit(1)


@dataclass
class RequestResult:
    """Result from a single request."""

    client_id: int
    request_id: int
    prompt: str
    max_tokens: int
    start_time: float
    first_token_time: Optional[float]
    end_time: float
    num_tokens: int
    success: bool
    error: Optional[str]
    ttft: Optional[float]
    tokens_per_sec: Optional[float]
    total_time: float


class LoadGenerator:
    """Load generator for benchmarking inference server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        num_clients: int = 10,
        requests_per_client: int = 10,
        use_streaming: bool = True,
        use_websocket: bool = False,
        prompt: str = "Hello, how are you?",
        max_tokens: int = 50,
    ):
        """Initialize load generator.

        Args:
            base_url: Base URL of the inference server
            num_clients: Number of concurrent clients
            requests_per_client: Number of requests per client
            use_streaming: Whether to use streaming endpoint
            use_websocket: Whether to use WebSocket (requires use_streaming=True)
            prompt: Prompt to send
            max_tokens: Maximum tokens to generate
        """
        self.base_url = base_url.rstrip("/")
        self.num_clients = num_clients
        self.requests_per_client = requests_per_client
        self.use_streaming = use_streaming
        self.use_websocket = use_websocket
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.results: List[RequestResult] = []

    def _make_sse_request(self, client_id: int, request_id: int) -> RequestResult:
        """Make a single SSE streaming request."""
        start_time = time.time()
        first_token_time = None
        num_tokens = 0
        success = False
        error = None

        try:
            url = f"{self.base_url}/v1/generate_stream"
            payload = {
                "prompt": self.prompt,
                "max_tokens": self.max_tokens,
                "temperature": 1.0,
            }

            with httpx.stream("POST", url, json=payload, timeout=300.0) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        try:
                            chunk = json.loads(data_str)

                            if "error" in chunk:
                                error = chunk["error"]
                                break

                            if chunk.get("token"):
                                num_tokens += 1
                                if first_token_time is None:
                                    first_token_time = time.time()

                            if chunk.get("finished"):
                                success = True
                                break
                        except json.JSONDecodeError:
                            continue

            end_time = time.time()
            total_time = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else None
            tokens_per_sec = num_tokens / total_time if total_time > 0 else None

        except Exception as e:
            end_time = time.time()
            error = str(e)
            total_time = end_time - start_time

        return RequestResult(
            client_id=client_id,
            request_id=request_id,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            start_time=start_time,
            first_token_time=first_token_time,
            end_time=end_time,
            num_tokens=num_tokens,
            success=success,
            error=error,
            ttft=ttft,
            tokens_per_sec=tokens_per_sec,
            total_time=total_time,
        )

    async def _make_websocket_request(
        self, client_id: int, request_id: int
    ) -> RequestResult:
        """Make a single WebSocket streaming request."""
        start_time = time.time()
        first_token_time = None
        num_tokens = 0
        success = False
        error = None

        try:
            ws_url = self.base_url.replace("http://", "ws://").replace(
                "https://", "wss://"
            )
            ws_url = f"{ws_url}/v1/generate_stream_ws"

            payload = {
                "prompt": self.prompt,
                "max_tokens": self.max_tokens,
                "temperature": 1.0,
            }

            async with websockets.connect(ws_url) as websocket:
                await websocket.send(json.dumps(payload))

                while True:
                    message = await websocket.recv()
                    chunk = json.loads(message)

                    if "error" in chunk:
                        error = chunk["error"]
                        break

                    if chunk.get("token"):
                        num_tokens += 1
                        if first_token_time is None:
                            first_token_time = time.time()

                    if chunk.get("finished"):
                        success = True
                        break

            end_time = time.time()
            total_time = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else None
            tokens_per_sec = num_tokens / total_time if total_time > 0 else None

        except Exception as e:
            end_time = time.time()
            error = str(e)
            total_time = end_time - start_time

        return RequestResult(
            client_id=client_id,
            request_id=request_id,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            start_time=start_time,
            first_token_time=first_token_time,
            end_time=end_time,
            num_tokens=num_tokens,
            success=success,
            error=error,
            ttft=ttft,
            tokens_per_sec=tokens_per_sec,
            total_time=total_time,
        )

    def _make_non_streaming_request(
        self, client_id: int, request_id: int
    ) -> RequestResult:
        """Make a single non-streaming request."""
        start_time = time.time()
        success = False
        error = None
        num_tokens = 0

        try:
            url = f"{self.base_url}/v1/generate"
            payload = {
                "prompt": self.prompt,
                "max_tokens": self.max_tokens,
                "temperature": 1.0,
            }

            response = httpx.post(url, json=payload, timeout=300.0)
            response.raise_for_status()
            result = response.json()

            num_tokens = result.get("num_tokens", 0)
            success = True

        except Exception as e:
            error = str(e)

        end_time = time.time()
        total_time = end_time - start_time

        return RequestResult(
            client_id=client_id,
            request_id=request_id,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            start_time=start_time,
            first_token_time=None,  # Not available for non-streaming
            end_time=end_time,
            num_tokens=num_tokens,
            success=success,
            error=error,
            ttft=None,
            tokens_per_sec=num_tokens / total_time if total_time > 0 else None,
            total_time=total_time,
        )

    def _client_worker(self, client_id: int) -> List[RequestResult]:
        """Worker function for a single client."""
        results = []
        for request_id in range(self.requests_per_client):
            if self.use_websocket:
                # Run async function in sync context
                result = asyncio.run(
                    self._make_websocket_request(client_id, request_id)
                )
            elif self.use_streaming:
                result = self._make_sse_request(client_id, request_id)
            else:
                result = self._make_non_streaming_request(client_id, request_id)

            results.append(result)
            print(
                f"Client {client_id}, Request {request_id}: {'✓' if result.success else '✗'} "
                f"(TTFT: {result.ttft:.3f}s, Tokens/sec: {result.tokens_per_sec:.2f})"
                if result.success and result.ttft
                else f"Client {client_id}, Request {request_id}: {'✓' if result.success else '✗'}"
            )

        return results

    def run(self) -> List[RequestResult]:
        """Run load generation."""
        print("Starting load generation:")
        print(f"  Clients: {self.num_clients}")
        print(f"  Requests per client: {self.requests_per_client}")
        print(f"  Total requests: {self.num_clients * self.requests_per_client}")
        print(f"  Streaming: {self.use_streaming}")
        print(f"  WebSocket: {self.use_websocket}")
        print(f"  Server: {self.base_url}")
        print()

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
            futures = [
                executor.submit(self._client_worker, client_id)
                for client_id in range(self.num_clients)
            ]

            for future in as_completed(futures):
                try:
                    results = future.result()
                    self.results.extend(results)
                except Exception as e:
                    print(f"Error in client worker: {e}")

        total_time = time.time() - start_time
        print(f"\nCompleted {len(self.results)} requests in {total_time:.2f}s")

        return self.results

    def save_csv(self, filename: str):
        """Save results to CSV file."""
        if not self.results:
            print("No results to save")
            return

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "client_id",
                    "request_id",
                    "prompt",
                    "max_tokens",
                    "start_time",
                    "first_token_time",
                    "end_time",
                    "num_tokens",
                    "success",
                    "error",
                    "ttft",
                    "tokens_per_sec",
                    "total_time",
                ]
            )

            for result in self.results:
                writer.writerow(
                    [
                        result.client_id,
                        result.request_id,
                        result.prompt,
                        result.max_tokens,
                        result.start_time,
                        result.first_token_time if result.first_token_time else "",
                        result.end_time,
                        result.num_tokens,
                        result.success,
                        result.error if result.error else "",
                        result.ttft if result.ttft else "",
                        result.tokens_per_sec if result.tokens_per_sec else "",
                        result.total_time,
                    ]
                )

        print(f"Results saved to {filename}")

        # Print summary statistics
        successful = [r for r in self.results if r.success]
        if successful:
            ttfts = [r.ttft for r in successful if r.ttft is not None]
            tps = [r.tokens_per_sec for r in successful if r.tokens_per_sec is not None]

            print("\nSummary Statistics:")
            print(f"  Successful requests: {len(successful)}/{len(self.results)}")
            if ttfts:
                print(
                    f"  TTFT - Mean: {sum(ttfts) / len(ttfts):.3f}s, Min: {min(ttfts):.3f}s, Max: {max(ttfts):.3f}s"
                )
            if tps:
                print(
                    f"  Tokens/sec - Mean: {sum(tps) / len(tps):.2f}, Min: {min(tps):.2f}, Max: {max(tps):.2f}"
                )


def main():
    parser = argparse.ArgumentParser(description="Load generator for inference server")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Server base URL"
    )
    parser.add_argument(
        "--clients", type=int, default=10, help="Number of concurrent clients"
    )
    parser.add_argument("--requests", type=int, default=10, help="Requests per client")
    parser.add_argument(
        "--prompt", default="Hello, how are you?", help="Prompt to send"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50, help="Max tokens to generate"
    )
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming")
    parser.add_argument(
        "--websocket", action="store_true", help="Use WebSocket instead of SSE"
    )
    parser.add_argument(
        "--output", default="loadgen_results.csv", help="Output CSV file"
    )

    args = parser.parse_args()

    generator = LoadGenerator(
        base_url=args.url,
        num_clients=args.clients,
        requests_per_client=args.requests,
        use_streaming=not args.no_streaming,
        use_websocket=args.websocket,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )

    generator.run()
    generator.save_csv(args.output)


if __name__ == "__main__":
    main()
