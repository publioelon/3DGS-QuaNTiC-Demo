# Thin wrapper around the Neural Transformation Cache.
#
# The NTC receives Gaussian positions, contracts them into the learned scene
# bounds, and predicts per-Gaussian motion for the current frame. Points outside
# the bounds are masked out and keep the identity/no-motion update. The actual
# network is the tiny-cuda-nn model loaded from each checkpoint.

import torch
class NeuralTransformationCache(torch.nn.Module):
    def __init__(self, model, xyz_bound_min, xyz_bound_max):
        super(NeuralTransformationCache, self).__init__()
        self.model = model
        self.register_buffer('xyz_bound_min',xyz_bound_min)
        self.register_buffer('xyz_bound_max',xyz_bound_max)
        
    def dump(self, path):
        torch.save(self.state_dict(),path)
        
    def get_contracted_xyz(self, xyz):
        with torch.no_grad():
            contracted_xyz=(xyz-self.xyz_bound_min)/(self.xyz_bound_max-self.xyz_bound_min)
            return contracted_xyz
        
    def forward(self, xyz:torch.Tensor):
        contracted_xyz=self.get_contracted_xyz(xyz)                          # Shape: [N, 3]
        
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)
        
        res_cache_inputs=torch.cat([contracted_xyz[mask]],dim=-1)
        resi=self.model(res_cache_inputs)
        
        masked_d_xyz=resi[:,:3]
        masked_d_rot=resi[:,3:7]
        # masked_d_opacity=resi[:,7:None]
        
        d_xyz = torch.full((xyz.shape[0], 3), 0.0, dtype=torch.half, device="cuda")
        d_rot = torch.full((xyz.shape[0], 4), 0.0, dtype=torch.half, device="cuda")
        d_rot[:, 0] = 1.0
        # d_opacity = self._origin_d_opacity.clone()

        d_xyz[mask] = masked_d_xyz
        d_rot[mask] = masked_d_rot
        
        return mask, d_xyz, d_rot

    @torch.no_grad()
    def forward_subset(self, xyz: torch.Tensor, indices: torch.Tensor):
        """Evaluate the NTC only for a preselected Gaussian subset.

        Unlike applying a post-hoc mask to forward(), this reduces the number of
        coordinates passed through the HashGrid + MLP. It is therefore the path
        used by compute-budgeted clients.

        Returns tensors in subset order:
            valid_subset_mask: [K]
            d_xyz_subset:      [K, 3]
            d_rot_subset:      [K, 4]

        Non-selected Gaussians are never evaluated here.
        """
        if indices is None:
            raise ValueError("indices must be provided")

        indices = indices.to(device=xyz.device, dtype=torch.long).reshape(-1)
        if indices.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.bool, device=xyz.device),
                torch.empty((0, 3), dtype=torch.half, device=xyz.device),
                torch.empty((0, 4), dtype=torch.half, device=xyz.device),
            )

        xyz_subset = xyz.index_select(0, indices)
        contracted_xyz = self.get_contracted_xyz(xyz_subset)

        valid = ((contracted_xyz >= 0) & (contracted_xyz <= 1)).all(dim=1)
        valid_inputs = contracted_xyz[valid]

        d_xyz = torch.zeros((indices.numel(), 3), dtype=torch.half, device=xyz.device)
        d_rot = torch.zeros((indices.numel(), 4), dtype=torch.half, device=xyz.device)
        d_rot[:, 0] = 1.0

        if valid_inputs.numel() != 0:
            resi = self.model(valid_inputs)
            d_xyz[valid] = resi[:, :3]
            d_rot[valid] = resi[:, 3:7]

        return valid, d_xyz, d_rot
        
        
