import torch

class CountSketch:
    def __init__(self, d, m, seed=None, num_rows: int = 4):
        """
        A multi-row Count-Sketch suitable for constant-size merges.

        Args:
            d: Dimension of the *original* vector to be sketched.
            m: Number of buckets (width) **per row**.
            seed: Optional RNG seed to make the hash/sign functions reproducible
                  across all ranks (so that sketches are additively mergeable).
            num_rows: Number of independent hash/sign pairs (i.e. rows).
                      A small constant (4-8) is usually sufficient.
        """

        self.d = d
        self.m = m
        self.r = num_rows  # number of rows

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Deterministic seeding so that every rank builds *identical* hash/sign tables.
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        # Hash functions h_j and sign functions s_j for every row.
        # Shapes: (r, d)
        self.h = torch.randint(0, m, (self.r, d), device=self.device, dtype=torch.int64)
        self.s = (torch.randint(0, 2, (self.r, d), device=self.device, dtype=torch.int64) * 2 - 1).to(torch.float32)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return an ``(r, m)`` Count-Sketch of the input vector ``x``.

        Args:
            x: 1-D tensor of length ``d``.
        Returns:
            Tensor of shape ``(r, m)`` where ``r`` is ``num_rows``.
        """
        assert x.shape[-1] == self.d, "Input dimension mismatch with CountSketch.d"

        # ``sketch[j]`` stores the histogram for row *j*.
        sketch = torch.zeros(self.r, self.m, device=self.device, dtype=torch.float32)

        # Iterate over rows (``r`` is a tiny constant, so this is negligible).
        for j in range(self.r):
            vals = self.s[j] * x  # signed values for this row
            sketch[j].scatter_add_(0, self.h[j], vals)

        return sketch

    def estimate(self, sketch: torch.Tensor, idx: torch.Tensor | int) -> torch.Tensor:
        """Median-of-rows estimate for ``x[idx]`` from a *merged* sketch.

        Args:
            sketch: Tensor of shape ``(r, m)`` – typically the AllReduced CMS.
            idx:    Scalar int or 1-D tensor of indices whose values we wish to estimate.

        Returns:
            Tensor of the same shape as ``idx`` with the CMS estimates.
        """

        if isinstance(idx, int):
            idx = torch.tensor([idx], device=self.device)
        else:
            idx = idx.to(self.device)

        per_row_estimates = []  # list of (len(idx),) tensors
        for j in range(self.r):
            est_j = sketch[j, self.h[j, idx]] * self.s[j, idx]
            per_row_estimates.append(est_j)

        stacked = torch.stack(per_row_estimates, dim=0)  # (r, len(idx))
        # Median across the ``r`` rows – robust against a few hash collisions.
        median = torch.median(stacked, dim=0).values

        return median if median.numel() > 1 else median.squeeze(0)

    def heavy_hitter_indices(self, sketch: torch.Tensor, k: int) -> torch.Tensor:
        """Return *approximate* top-``k`` heavy-hitter indices via CMS scan."""

        # Estimate every coordinate in one shot on-device (O(d) but local).
        all_idx = torch.arange(self.d, device=self.device)
        estimates = self.estimate(sketch, all_idx)

        # Top-k by magnitude.
        _, topk_idx = torch.topk(estimates.abs(), k)
        return topk_idx
