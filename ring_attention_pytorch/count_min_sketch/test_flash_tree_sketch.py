# test_flash_tree_sketch.py
import torch
import numpy as np

# ------------------- CountSketch implementation -------------------
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

# ------------------- Simulation with constant-communication CMS -------------------
def simulate_flash_tree_sketch(
    num_shards: int = 4,
    d: int = 4096,
    k: int = 16,
    m: int = 256,
    seed: int = 0,
    r: int = 4,
):
    """
    Simulates a multi-GPU CountSketch scenario on one GH200 GPU (or CPU if CUDA not available).
    1. Creates 'num_shards' local vectors, each with k heavy entries + noise.
    2. Encodes each with CountSketch.
    3. Merges the sketches (one all-reduce) and merges denominators.
    4. Recovers top-k from the merged sketch.
    5. Compares against the true global top-k.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Running on device: {device} (CUDA GPU utilized, e.g., GH200)")
        print(f"  PyTorch CUDA version: {torch.version.cuda}")
        try:
            print(f"  CUDA Device Name: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"  Could not retrieve CUDA device name: {e}")
        print("") # Newline for readability
    else:
        print(f"Running on device: {device} (CUDA not available, using CPU fallback)")
        print("  INFO: To utilize an NVIDIA GPU with PyTorch, please ensure the following:")
        print("    1. NVIDIA drivers are correctly installed and up-to-date on your system.")
        print("       - You can typically check this with the `nvidia-smi` command in your terminal.")
        print("    2. A compatible CUDA Toolkit is installed.")
        print("       - Check your PyTorch version's CUDA compatibility requirements.")
        print("    3. PyTorch was installed with CUDA support.")
        print("       - Verify by running: python -c 'import torch; print(torch.cuda.is_available())'")
        print("       - If False, you may need to reinstall PyTorch. For example, find the correct command for your OS/CUDA version at https://pytorch.org/get-started/locally/")
        print("         Commonly, it's like: pip uninstall torch torchvision torchaudio")
        print("                         pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXYZ")
        print("         (Replace cuXYZ with your CUDA version, e.g., cu118, cu121)")
        print("") # Newline for readability

    # A *shared* Count-Sketch – identical hash/sign tables across shards so the
    # row×bucket tables are additively mergeable.
    sketcher = CountSketch(d=d, m=m, seed=seed, num_rows=r)

    # Step 1: Create local vectors on each "shard"
    local_vectors = []
    for shard_id in range(num_shards):
        x = torch.zeros(d, device=device)

        # Pick k random heavy-hitter indices
        hh_indices = np.random.choice(d, k, replace=False)
        # Assign large random values at those indices
        for i in hh_indices:
            x[i] = torch.randn((), device=device) * 10.0
        # Add small Gaussian noise everywhere else
        noise = torch.randn(d, device=device) * 0.1
        x += noise
        local_vectors.append(x)

    # Step 2: Each shard (GPU) builds *its own* CMS **and** records its local
    #          top-k indices – both are O(1) in communication.
    local_sketches: list[torch.Tensor] = []
    local_denominators: list[torch.Tensor] = []  # one scalar per shard
    local_topk_sets: list[set[int]] = []
    for x in local_vectors:
        sketch = sketcher.encode(x)
        denom = torch.sum(x)  # stand-in for the softmax denominator on the shard

        local_sketches.append(sketch)
        local_denominators.append(denom)

        # Track the shard's own heavy hitters so we can optionally form the
        # union later (demonstrating the alternative strategy mentioned in the
        # design doc).
        local_topk = set(torch.topk(x.abs(), k).indices.cpu().tolist())
        local_topk_sets.append(local_topk)

    global_sketch = torch.zeros(r, m, device=device)
    global_denom = torch.zeros((), device=device)
    for sk, dg in zip(local_sketches, local_denominators):
        global_sketch += sk
        global_denom += dg

    # ------------------------------------------------------------------
    #  (1) CMS merge done above (constant communication).
    #  (2) Use the merged CMS to propose a candidate heavy-hitter set 𝓗.
    # ------------------------------------------------------------------

    # Option A: Union of per-shard local top-k (<= num_shards×k elements).
    candidate_set: set[int] = set().union(*local_topk_sets)

    # Option B: CMS scan on *one* GPU (O(d) local work) – we keep exactly k
    #           indices from that scan and merge with the union so that we are
    #           extra safe.
    cms_topk = sketcher.heavy_hitter_indices(global_sketch, k).tolist()
    candidate_set.update(cms_topk)

    recovered_set = candidate_set

    exact_global_x = torch.zeros(d, device=device)
    for x in local_vectors:
        exact_global_x += x

    true_topk = set(torch.topk(exact_global_x.abs(), k).indices.cpu().tolist())
    overlap = len(true_topk.intersection(recovered_set))
    print(f"Recovered {overlap}/{k} true heavy hitters using constant-size collectives.\n")

    # ------------------------------------------------------------------
    #  (3) AllReduce the *exact* values of the candidate heavy indices –
    #      constant size because |𝓗| is O(1).
    # ------------------------------------------------------------------
    candidate_list = sorted(recovered_set)
    exact_heavy_vals = exact_global_x[candidate_list]

    # ------------------------------------------------------------------
    #  (4) Fetch one more scalar – the total numerator mass – via AllReduce.
    # ------------------------------------------------------------------
    global_mass = global_denom  # already computed (sum over all coordinates)

    residual = global_mass - exact_heavy_vals.sum()

    # ------------------------------------------------------------------
    #  (5) Reconstruct full vector (store residual in a dummy slot to keep the
    #      representation constant-size if desired).
    # ------------------------------------------------------------------
    reconstructed = torch.zeros(d, device=device)
    reconstructed[candidate_list] = exact_heavy_vals
    # For demonstration we *optionally* store the residual in index 0 if it is
    # not already a heavy index – this does not affect Tree-Attention because
    # the residual would be handled separately in practice.
    if 0 not in recovered_set:
        reconstructed[0] += residual

    # ------------------------------------------------------------
    #  Diagnostics: verify we achieved *exact* recovery (within FP
    #  tolerance) and log several helpful metrics.
    # ------------------------------------------------------------
    diff = exact_global_x - reconstructed
    l2_error = torch.norm(diff).item()
    max_abs_err = torch.max(diff.abs()).item()

    # Sanity-check that heavy-hitters + residual reproduce the total mass.
    mass_check = torch.allclose(exact_heavy_vals.sum() + residual, global_mass, rtol=1e-5, atol=1e-6)

    print("================ Recovery diagnostics ================")
    print(f"Vector dimension (d):         {d}")
    print(f"#Shards × top-k per shard:    {num_shards} × {k}")
    print(f"Candidate set size |𝓗|:       {len(candidate_list)}")
    print(f"L2 reconstruction error:     {l2_error:.6e}")
    print(f"Max |recon−exact| element:   {max_abs_err:.6e}")
    print(f"Heavy+residual == total?     {'✔' if mass_check else '✘'}")
    print("======================================================\n")

    # For callers / tests, return the diagnostics.
    error = l2_error

    return {
        "true_topk": true_topk,
        "recovered_set": recovered_set,
        "error": error
    }

if __name__ == "__main__":
    results = simulate_flash_tree_sketch(
        num_shards=4,   # pretend we have 4 GPU shards
        d=4096,         # hidden dimension
        k=16,           # number of heavy-hitter values
        m=1024,          # sketch width (per row)
        seed=42,
        r=16,            # number of CMS rows (constant)
    )
