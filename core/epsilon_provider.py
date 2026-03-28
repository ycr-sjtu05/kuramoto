import torch


class BaseEpsilonProvider:
    requires_ids = False

    def __call__(self, sample_ids, shape, device, dtype):
        raise NotImplementedError


class FreshEpsilonProvider(BaseEpsilonProvider):
    requires_ids = False

    def __call__(self, sample_ids, shape, device, dtype):
        return torch.randn(shape, device=device, dtype=dtype)


class FixedPerSampleEpsilonProvider(BaseEpsilonProvider):
    """
    为每个样本 idx 生成一份固定 epsilon。
    不需要额外落盘，直接由 (base_seed + sample_id) 决定。
    """
    requires_ids = True

    def __init__(self, base_seed=12345):
        self.base_seed = int(base_seed)

    def _seed_from_id(self, sample_id: int) -> int:
        # 简单稳定即可；后续你要更复杂 hash 再换
        return self.base_seed + int(sample_id)

    def __call__(self, sample_ids, shape, device, dtype):
        if sample_ids is None:
            raise ValueError("FixedPerSampleEpsilonProvider 需要 sample_ids，但收到了 None。")

        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.detach().cpu().tolist()

        B, C, H, W = shape
        eps_list = []

        for sid in sample_ids:
            g = torch.Generator(device="cpu")
            g.manual_seed(self._seed_from_id(sid))
            eps_i = torch.randn((C, H, W), generator=g, dtype=torch.float32)
            eps_list.append(eps_i)

        eps = torch.stack(eps_list, dim=0).to(device=device, dtype=dtype)
        return eps


def build_epsilon_provider(mode="fresh", base_seed=12345):
    if mode == "fresh":
        return FreshEpsilonProvider()
    if mode == "fixed_per_sample":
        return FixedPerSampleEpsilonProvider(base_seed=base_seed)

    raise ValueError(f"不支持的 epsilon_mode: {mode}")