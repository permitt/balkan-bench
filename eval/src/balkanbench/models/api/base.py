"""API-backed model runner: concurrent fan-out, retry, on-disk cache, cost tracking.

``APIModel`` implements :class:`~balkanbench.models.generative_base.GenerativeModel`
against any provider reachable through a :class:`ProviderClient` - a thin
``complete(prompt, ...) -> APIResponse`` transport. This module is provider-
agnostic and imports no provider SDK: the anthropic/openai/gemini clients are
wired up in a later task and injected here via the ``client`` constructor
argument.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from balkanbench.models.generative_base import UnsupportedProtocolError

# Retry policy: up to 5 attempts total, exponential backoff between attempts
# (1s, 2s, 4s, 8s before the 2nd..5th attempt) plus jitter; re-raise the
# triggering exception verbatim after the 5th attempt fails.
_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0)
_JITTER_FRACTION = 0.1


@dataclass
class APIResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@runtime_checkable
class ProviderClient(Protocol):
    """Transport a real provider client (Task 9) implements against its SDK."""

    def complete(
        self, prompt: str, *, max_tokens: int, stop_sequences: list[str]
    ) -> APIResponse: ...


class APIModel:
    """Concurrent, cached, cost-tracked runner over a :class:`ProviderClient`.

    Reads from ``model_cfg``:
      - ``api_model_id`` (required): identifies the model to the provider and
        is folded into the cache key so different models never share entries.
      - ``generation.concurrency`` (default 4): ``ThreadPoolExecutor`` worker
        count for fanning ``generate()`` out over prompts.
      - ``generation.max_tokens`` (optional): a hard cap on generated tokens;
        each ``generate()`` call's ``max_gen_tokens`` is clamped to it.

    ``client`` is the injected :class:`ProviderClient` transport. ``None`` is
    accepted so a model factory can construct an ``APIModel`` before a real
    provider client exists and wire one in later; calling ``generate()``
    without ever setting a client raises ``RuntimeError``.
    """

    def __init__(
        self,
        model_cfg: dict[str, Any],
        *,
        cache_dir: Path,
        client: ProviderClient | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.api_model_id = model_cfg["api_model_id"]
        gen_cfg = model_cfg.get("generation", {})
        self.concurrency = int(gen_cfg.get("concurrency", 4))
        self._max_tokens_cap = gen_cfg.get("max_tokens")
        self.cache_dir = Path(cache_dir)
        self.client = client

        self._lock = threading.Lock()
        self._total_cost_usd = 0.0
        self._request_count = 0

    # -- GenerativeModel protocol ---------------------------------------------

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[float]:
        raise UnsupportedProtocolError(
            "APIModel has no notion of token-level logprobs; "
            "loglikelihood-based tasks are unsupported for API-backed models."
        )

    def generate(
        self,
        prompts: list[str],
        *,
        stop_sequences: list[str],
        max_gen_tokens: int,
    ) -> list[str]:
        """Greedy completion per prompt, fanned out over a thread pool.

        Returns results in the same order as ``prompts``, regardless of which
        request completes first: ``ThreadPoolExecutor.map`` yields results in
        call order, so no explicit re-sorting is required.
        """
        if self.client is None:
            raise RuntimeError("APIModel.client must be set before generate() is called")

        max_tokens = max_gen_tokens
        if self._max_tokens_cap is not None:
            max_tokens = min(max_tokens, int(self._max_tokens_cap))

        def call_one(prompt: str) -> str:
            response = self._complete_cached(prompt, max_tokens=max_tokens, stop=stop_sequences)
            return response.text

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(call_one, prompts))

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return self._total_cost_usd

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    # -- cache -----------------------------------------------------------------

    def _cache_key(self, prompt: str, *, max_tokens: int, stop: list[str]) -> str:
        payload = json.dumps(
            {
                "model": self.api_model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stop": stop,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, path: Path) -> APIResponse | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        try:
            return APIResponse(**data)
        except TypeError:
            return None

    def _write_cache(self, path: Path, response: APIResponse) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a uniquely-named temp file in the same directory, then
        # rename into place: os.replace is atomic on both POSIX and Windows,
        # so a crash or a concurrent writer for the same key never leaves a
        # partially-written or interleaved cache file at the final path.
        tmp_path = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(asdict(response)), encoding="utf-8")
        os.replace(tmp_path, path)

    def _complete_cached(self, prompt: str, *, max_tokens: int, stop: list[str]) -> APIResponse:
        key = self._cache_key(prompt, max_tokens=max_tokens, stop=stop)
        path = self._cache_path(key)

        cached = self._read_cache(path)
        if cached is not None:
            return cached

        response = self._complete_with_retry(prompt, max_tokens=max_tokens, stop=stop)
        self._write_cache(path, response)
        with self._lock:
            self._total_cost_usd += response.cost_usd
            self._request_count += 1
        return response

    # -- retry -------------------------------------------------------------------

    def _complete_with_retry(self, prompt: str, *, max_tokens: int, stop: list[str]) -> APIResponse:
        assert self.client is not None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return self.client.complete(prompt, max_tokens=max_tokens, stop_sequences=stop)
            except Exception:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                base_delay = _BACKOFF_BASE_SECONDS[attempt]
                time.sleep(base_delay + random.uniform(0, base_delay * _JITTER_FRACTION))
        raise AssertionError("unreachable")  # pragma: no cover
