from __future__ import annotations
from dataclasses import dataclass
import json
import csv
from pathlib import Path
from typing import TYPE_CHECKING

from sim.core.request import Request

if TYPE_CHECKING:
    pass


@dataclass
class TraceFormat:
    VIDUR = "vidur"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"


class TraceWorkloadLoader:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self._request_counter = 0

    def load(self, path: str, format: str = "auto") -> list[Request]:
        file_path = Path(path)

        if format == "auto":
            format = self._detect_format(file_path)

        if format == TraceFormat.VIDUR:
            return self._load_vidur(file_path)
        elif format == TraceFormat.SHAREGPT:
            return self._load_sharegpt(file_path)
        elif format == TraceFormat.CUSTOM:
            return self._load_custom(file_path)
        else:
            raise ValueError(f"Unknown trace format: {format}")

    def _detect_format(self, path: Path) -> str:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return TraceFormat.VIDUR
        elif suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    if "conversations" in data[0]:
                        return TraceFormat.SHAREGPT
            return TraceFormat.CUSTOM
        return TraceFormat.CUSTOM

    def _load_vidur(self, path: Path) -> list[Request]:
        requests = []

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt_len = int(row.get("prompt_len", row.get("input_tokens", 512)))
                output_len = int(row.get("output_len", row.get("output_tokens", 128)))
                arrival_time = float(row.get("arrival_time", 0.0))

                prompt_tokens = self._generate_dummy_tokens(prompt_len)

                request = Request(
                    id=self._request_counter,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=output_len,
                    arrival_time=arrival_time,
                )
                requests.append(request)
                self._request_counter += 1

        return requests

    def _load_sharegpt(self, path: Path) -> list[Request]:
        requests = []

        with open(path, "r") as f:
            data = json.load(f)

        current_time = 0.0
        inter_arrival = 0.1

        for item in data:
            conversations = item.get("conversations", [])

            prompt_len = 0
            output_len = 0

            for conv in conversations:
                if conv.get("from") == "human":
                    prompt_len += len(conv.get("value", "").split())
                elif conv.get("from") == "gpt":
                    output_len += len(conv.get("value", "").split())

            prompt_len = max(1, int(prompt_len * 1.3))
            output_len = max(1, int(output_len * 1.3))

            prompt_tokens = self._generate_dummy_tokens(prompt_len)

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_len,
                arrival_time=current_time,
            )
            requests.append(request)
            self._request_counter += 1
            current_time += inter_arrival

        return requests

    def _load_custom(self, path: Path) -> list[Request]:
        requests = []

        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("requests", [])

        for item in data:
            prompt_tokens = item.get("prompt_tokens")
            if prompt_tokens is None:
                prompt_len = item.get("prompt_len", 512)
                prompt_tokens = self._generate_dummy_tokens(prompt_len)

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=item.get("max_new_tokens", item.get("output_len", 128)),
                arrival_time=item.get("arrival_time", 0.0),
                stream=item.get("stream", False),
                prefix_group_id=item.get("prefix_group_id"),
            )
            requests.append(request)
            self._request_counter += 1

        return requests

    def _generate_dummy_tokens(self, length: int) -> list[int]:
        import random
        return [random.randint(0, self.vocab_size - 1) for _ in range(length)]

    def save_requests(self, requests: list[Request], path: str) -> None:
        data = {
            "requests": [
                {
                    "id": r.id,
                    "prompt_len": r.prompt_len,
                    "max_new_tokens": r.max_new_tokens,
                    "arrival_time": r.arrival_time,
                    "stream": r.stream,
                    "prefix_group_id": r.prefix_group_id,
                }
                for r in requests
            ]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

