from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from typing import TYPE_CHECKING

from sim.core.request import Request
from sim.workload.prefix_groups import PrefixGroupManager

if TYPE_CHECKING:
    pass


class DistributionType(Enum):
    CONSTANT = "constant"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    ZIPF = "zipf"


@dataclass
class Distribution:
    type: DistributionType
    params: dict = field(default_factory=dict)

    def sample(self) -> float:
        if self.type == DistributionType.CONSTANT:
            return self.params.get("value", 0.0)

        elif self.type == DistributionType.UNIFORM:
            low = self.params.get("low", 0.0)
            high = self.params.get("high", 1.0)
            return random.uniform(low, high)

        elif self.type == DistributionType.EXPONENTIAL:
            rate = self.params.get("rate", 1.0)
            return random.expovariate(rate)

        elif self.type == DistributionType.NORMAL:
            mean = self.params.get("mean", 0.0)
            std = self.params.get("std", 1.0)
            return random.gauss(mean, std)

        elif self.type == DistributionType.LOGNORMAL:
            mean = self.params.get("mean", 0.0)
            sigma = self.params.get("sigma", 1.0)
            return random.lognormvariate(mean, sigma)

        elif self.type == DistributionType.ZIPF:
            alpha = self.params.get("alpha", 1.5)
            n = self.params.get("n", 1000)
            return self._zipf_sample(alpha, n)

        return 0.0

    def sample_int(self, min_val: int = 1, max_val: int | None = None) -> int:
        value = int(self.sample())
        value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
        return value

    def _zipf_sample(self, alpha: float, n: int) -> float:
        harmonic = sum(1.0 / (i ** alpha) for i in range(1, n + 1))
        u = random.random()
        cumulative = 0.0
        for i in range(1, n + 1):
            cumulative += 1.0 / (i ** alpha) / harmonic
            if u <= cumulative:
                return float(i)
        return float(n)

    @classmethod
    def constant(cls, value: float) -> Distribution:
        return cls(DistributionType.CONSTANT, {"value": value})

    @classmethod
    def uniform(cls, low: float, high: float) -> Distribution:
        return cls(DistributionType.UNIFORM, {"low": low, "high": high})

    @classmethod
    def exponential(cls, rate: float) -> Distribution:
        return cls(DistributionType.EXPONENTIAL, {"rate": rate})

    @classmethod
    def normal(cls, mean: float, std: float) -> Distribution:
        return cls(DistributionType.NORMAL, {"mean": mean, "std": std})

    @classmethod
    def lognormal(cls, mean: float, sigma: float) -> Distribution:
        return cls(DistributionType.LOGNORMAL, {"mean": mean, "sigma": sigma})


class SyntheticWorkloadGenerator:
    def __init__(
        self,
        arrival_rate: float,
        prompt_len_dist: Distribution,
        output_len_dist: Distribution,
        prefix_groups: PrefixGroupManager | None = None,
        vocab_size: int = 32000,
        seed: int | None = None,
    ):
        self.arrival_rate = arrival_rate
        self.prompt_len_dist = prompt_len_dist
        self.output_len_dist = output_len_dist
        self.prefix_groups = prefix_groups
        self.vocab_size = vocab_size

        if seed is not None:
            random.seed(seed)

        self._request_counter = 0

    def generate(self, duration: float) -> list[Request]:
        requests: list[Request] = []
        current_time = 0.0

        while current_time < duration:
            inter_arrival = random.expovariate(self.arrival_rate)
            current_time += inter_arrival

            if current_time >= duration:
                break

            prompt_len = self.prompt_len_dist.sample_int(min_val=1, max_val=32768)
            output_len = self.output_len_dist.sample_int(min_val=1, max_val=4096)

            if self.prefix_groups is not None:
                prompt_tokens, prefix_group_id = self.prefix_groups.generate_prompt(prompt_len)
            else:
                prompt_tokens = [
                    random.randint(0, self.vocab_size - 1) for _ in range(prompt_len)
                ]
                prefix_group_id = None

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_len,
                arrival_time=current_time,
                prefix_group_id=prefix_group_id,
            )
            requests.append(request)
            self._request_counter += 1

        return requests

    def generate_batch(self, num_requests: int, start_time: float = 0.0) -> list[Request]:
        requests: list[Request] = []
        current_time = start_time

        for _ in range(num_requests):
            inter_arrival = random.expovariate(self.arrival_rate)
            current_time += inter_arrival

            prompt_len = self.prompt_len_dist.sample_int(min_val=1, max_val=32768)
            output_len = self.output_len_dist.sample_int(min_val=1, max_val=4096)

            if self.prefix_groups is not None:
                prompt_tokens, prefix_group_id = self.prefix_groups.generate_prompt(prompt_len)
            else:
                prompt_tokens = [
                    random.randint(0, self.vocab_size - 1) for _ in range(prompt_len)
                ]
                prefix_group_id = None

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_len,
                arrival_time=current_time,
                prefix_group_id=prefix_group_id,
            )
            requests.append(request)
            self._request_counter += 1

        return requests

    @classmethod
    def default(cls, arrival_rate: float = 10.0) -> SyntheticWorkloadGenerator:
        return cls(
            arrival_rate=arrival_rate,
            prompt_len_dist=Distribution.uniform(256, 2048),
            output_len_dist=Distribution.uniform(64, 512),
        )

    @classmethod
    def chat_workload(cls, arrival_rate: float = 5.0) -> SyntheticWorkloadGenerator:
        return cls(
            arrival_rate=arrival_rate,
            prompt_len_dist=Distribution.lognormal(mean=6.0, sigma=1.0),  # ~400 avg
            output_len_dist=Distribution.lognormal(mean=5.0, sigma=1.0),  # ~150 avg
        )

    @classmethod
    def batch_inference(cls, num_requests: int = 1000) -> SyntheticWorkloadGenerator:
        return cls(
            arrival_rate=float("inf"),
            prompt_len_dist=Distribution.uniform(512, 4096),
            output_len_dist=Distribution.uniform(128, 1024),
        )

