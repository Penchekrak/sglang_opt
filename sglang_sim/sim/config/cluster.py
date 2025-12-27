from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class InterconnectType(Enum):
    NVLINK = "nvlink"
    ETHERNET = "ethernet"
    INFINIBAND = "infiniband"


@dataclass
class InterconnectConfig:
    type: InterconnectType
    bandwidth_gbps: float
    latency_us: float

    @property
    def bandwidth_bytes_per_sec(self) -> float:
        return self.bandwidth_gbps * 1e9 / 8

    @property
    def latency_seconds(self) -> float:
        return self.latency_us * 1e-6

    def transfer_time_seconds(self, bytes_to_transfer: int) -> float:
        return self.latency_seconds + bytes_to_transfer / self.bandwidth_bytes_per_sec

    @classmethod
    def nvlink_4(cls) -> InterconnectConfig:
        return cls(
            type=InterconnectType.NVLINK,
            bandwidth_gbps=900,  # NVLink 4.0: ~900 Gbps bidirectional
            latency_us=1.0,
        )

    @classmethod
    def infiniband_hdr(cls) -> InterconnectConfig:
        return cls(
            type=InterconnectType.INFINIBAND,
            bandwidth_gbps=200,  # HDR InfiniBand
            latency_us=2.0,
        )

    @classmethod
    def ethernet_100g(cls) -> InterconnectConfig:
        return cls(
            type=InterconnectType.ETHERNET,
            bandwidth_gbps=100,
            latency_us=10.0,
        )


@dataclass
class GPUSpec:
    name: str
    flops_fp16: float  # FLOP/s
    memory_bandwidth: float  # bytes/s
    memory_capacity: int  # bytes

    @classmethod
    def h100_sxm(cls) -> GPUSpec:
        return cls(
            name="H100-SXM",
            flops_fp16=1979e12,  # ~2 PFLOP/s FP16
            memory_bandwidth=3.35e12,  # 3.35 TB/s
            memory_capacity=80 * 1024**3,  # 80 GB
        )

    @classmethod
    def a100_80g(cls) -> GPUSpec:
        return cls(
            name="A100-80G",
            flops_fp16=312e12,  # 312 TFLOP/s FP16
            memory_bandwidth=2.0e12,  # 2 TB/s
            memory_capacity=80 * 1024**3,
        )

    @classmethod
    def h200(cls) -> GPUSpec:
        return cls(
            name="H200",
            flops_fp16=1979e12,
            memory_bandwidth=4.8e12,  # 4.8 TB/s with HBM3e
            memory_capacity=141 * 1024**3,  # 141 GB
        )


@dataclass
class ClusterConfig:
    num_prefill_workers: int
    num_decode_workers: int
    gpus_per_worker: int
    gpu_spec: GPUSpec
    intra_node_interconnect: InterconnectConfig
    inter_node_interconnect: InterconnectConfig

    @property
    def total_workers(self) -> int:
        return self.num_prefill_workers + self.num_decode_workers

    @property
    def total_gpus(self) -> int:
        return self.total_workers * self.gpus_per_worker

    @classmethod
    def single_node_8gpu(cls) -> ClusterConfig:
        return cls(
            num_prefill_workers=4,
            num_decode_workers=4,
            gpus_per_worker=1,
            gpu_spec=GPUSpec.h100_sxm(),
            intra_node_interconnect=InterconnectConfig.nvlink_4(),
            inter_node_interconnect=InterconnectConfig.nvlink_4(),
        )

    @classmethod
    def multi_node_16gpu(cls) -> ClusterConfig:
        return cls(
            num_prefill_workers=8,
            num_decode_workers=8,
            gpus_per_worker=1,
            gpu_spec=GPUSpec.h100_sxm(),
            intra_node_interconnect=InterconnectConfig.nvlink_4(),
            inter_node_interconnect=InterconnectConfig.infiniband_hdr(),
        )

