from __future__ import annotations
from dataclasses import dataclass, field
import random


@dataclass
class PrefixGroup:
    prefix_tokens: list[int]
    share_ratio: float
    group_id: int = 0

    @property
    def prefix_length(self) -> int:
        return len(self.prefix_tokens)

    def sample_tokens(self, additional_length: int, vocab_size: int = 32000) -> list[int]:
        suffix = [random.randint(0, vocab_size - 1) for _ in range(additional_length)]
        return self.prefix_tokens + suffix


@dataclass
class PrefixGroupManager:
    groups: list[PrefixGroup] = field(default_factory=list)
    vocab_size: int = 32000

    def add_group(
        self,
        prefix_length: int,
        share_ratio: float,
        group_id: int | None = None,
    ) -> PrefixGroup:
        if group_id is None:
            group_id = len(self.groups)

        prefix_tokens = [random.randint(0, self.vocab_size - 1) for _ in range(prefix_length)]
        group = PrefixGroup(
            prefix_tokens=prefix_tokens,
            share_ratio=share_ratio,
            group_id=group_id,
        )
        self.groups.append(group)
        return group

    def sample_group(self) -> PrefixGroup | None:
        if not self.groups:
            return None

        total_ratio = sum(g.share_ratio for g in self.groups)
        if total_ratio == 0:
            return None

        r = random.random() * total_ratio
        cumulative = 0.0
        for group in self.groups:
            cumulative += group.share_ratio
            if r <= cumulative:
                return group

        return self.groups[-1] if self.groups else None

    def generate_prompt(self, total_length: int) -> tuple[list[int], int | None]:
        group = self.sample_group()
        if group is None:
            tokens = [random.randint(0, self.vocab_size - 1) for _ in range(total_length)]
            return tokens, None

        if group.prefix_length >= total_length:
            return group.prefix_tokens[:total_length], group.group_id

        additional = total_length - group.prefix_length
        tokens = group.sample_tokens(additional, self.vocab_size)
        return tokens, group.group_id

    @classmethod
    def create_uniform_groups(
        cls,
        num_groups: int,
        prefix_length: int,
        vocab_size: int = 32000,
    ) -> PrefixGroupManager:
        manager = cls(vocab_size=vocab_size)
        share_per_group = 1.0 / num_groups
        for i in range(num_groups):
            manager.add_group(prefix_length, share_per_group, group_id=i)
        return manager

    @classmethod
    def create_zipf_groups(
        cls,
        num_groups: int,
        prefix_length: int,
        zipf_param: float = 1.0,
        vocab_size: int = 32000,
    ) -> PrefixGroupManager:
        manager = cls(vocab_size=vocab_size)

        weights = [1.0 / (i + 1) ** zipf_param for i in range(num_groups)]
        total_weight = sum(weights)
        normalized = [w / total_weight for w in weights]

        for i, ratio in enumerate(normalized):
            manager.add_group(prefix_length, ratio, group_id=i)

        return manager

