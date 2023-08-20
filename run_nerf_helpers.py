import os

import cv2
import numpy as np

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255*np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



class MultipleModels(nn.Module):
    def __init__(self,
                 height_img_path: os.PathLike,
                 object_height: float,
                 n_models: int,
                 virtual_size: float,
                 threshold: int = 70,
                 D: int = 8,
                 W: int = 256,
                 input_ch: int = 3,
                 input_ch_views: int = 3,
                 output_ch: int = 4,
                 skips: list = [4],
                 use_viewdirs: bool = False,
                 device: str = "cpu",
                 multires: int = 8,
                 i_embed: int = 3,
                 multires_views: int = 3) -> None:
        super(MultipleModels, self).__init__()

        # variables for clustering
        self.heights_img_path = height_img_path
        self.object_heights = object_height
        self.threshold = threshold
        self.n_models = n_models
        self.virtual_size = virtual_size
        
        self.heights_img = self.get_cluster_map().to(device)

        # variables for nerf
        self.multires = multires
        self.i_embed = i_embed
        self.multires_views = multires_views
        self.use_viewdirs = use_viewdirs
        self.device = device

        self.models = nn.ModuleList([self.get_model(D=D, W=W, input_ch=input_ch,
                                                    output_ch=output_ch, skips=skips,
                                                    input_ch_views=input_ch_views,
                                                    use_viewdirs=use_viewdirs
                                                    ) for _ in range(self.n_models)])
        
        self.NeRF = self.get_model(D=D, W=W, input_ch=input_ch,
                                   output_ch=output_ch, skips=skips,
                                   input_ch_views=input_ch_views,
                                   use_viewdirs=use_viewdirs)

    def get_cluster_map(self):
        img = cv2.imread(self.heights_img_path, 0)
        self.resolution = img.shape[0]
        ind = np.where(img < self.threshold)
        X = np.array([ind[0], ind[1]]).T
        km = KMeans(n_clusters=self.n_models - 1,
                    init='random',
                    n_init=10,
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)
        y_km = km.fit_predict(X)
        img = np.where(img < self.threshold, img, 0)
        np.put(img, (ind[0]) * img.shape[0] + ind[1], y_km + 1)
        return torch.from_numpy(img)

    def adaptive_interval_sampling(self, rays_o, rays_d, z_vals, N_samples):
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        x, y = pts[..., 1].to(torch.long), pts[..., 0].to(torch.long)
        batch_indexes = self.heights_img[x, y]
        print("batch indexs: ", batch_indexes.shape)
        return None

    def get_model(self, D, W, input_ch, output_ch, skips,
                  input_ch_views, use_viewdirs, model_type='nerf'):
        if model_type == 'nerf':
            model = NeRF(D=D, W=W, input_ch=input_ch, output_ch=output_ch,
                         skips=skips, input_ch_views=input_ch_views,
                         use_viewdirs=use_viewdirs).to(self.device)
        else:
            raise NotImplementedError

        return model

    def get_indexes(self, inputs_flat):
        # inputs_flat = inputs_flat.cpu()
        x, y = (inputs_flat[:, 1] + self.virtual_size) / \
               (self.virtual_size * 2) * self.resolution, \
               (inputs_flat[:, 0] + self.virtual_size) / \
               (self.virtual_size * 2) * self.resolution
        x, y = x.to(torch.long), y.to(torch.long)
        # image = self.heights_img.cpu()
        batch_indexes = self.heights_img[x, y]
        return batch_indexes

    def forward(self, inputs_flat, input_dirs_flat):
        embed_fn, input_ch = get_embedder(self.multires, self.i_embed)
        embedded = embed_fn(inputs_flat)

        input_ch_views = 0
        embeddirs_fn = None
        if self.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(self.multires_views,
                                                        self.i_embed)
            embedded_dirs = embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        batch_indexes = self.get_indexes(inputs_flat)

        outputs = None
        for i in range(self.n_models):
            model_indexes = torch.where(batch_indexes == i, i, 0.).unsqueeze(0)
            model_indexes = torch.cat([model_indexes for _ in range(4)], 0).T
            # print(self.models[i])
            model = self.models[i]
            tmp_outputs = model(embedded)
            tmp_outputs = torch.mul(model_indexes, tmp_outputs)
            if outputs is None:
                outputs = tmp_outputs
            else:
                outputs += tmp_outputs
        return outputs


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


class AdaptiveIntervalSampling:
    """Adaptive Interval Sampling"""
    def __init__(self,
                 height_img_path: os.PathLike,
                 object_height: float,
                 n_models: int,
                 virtual_size: float,
                 threshold: int = 70,
                 device: str = "cpu") -> None:

        self.heights_img_path = height_img_path
        self.object_heights = object_height
        self.threshold = threshold
        self.n_models = n_models
        self.virtual_size = virtual_size

        # height_img to gpu
        self.heights_img = self.get_cluster_map().to(device)

    def get_cluster_map(self):
        img = cv2.imread(self.heights_img_path, 0)
        self.resolution = img.shape[0]
        ind = np.where(img < self.threshold)
        X = np.array([ind[0], ind[1]]).T
        km = KMeans(n_clusters=self.n_models,
                    init='random',
                    n_init=10,
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)
        y_km = km.fit_predict(X)
        img = np.where(img < self.threshold, img, 0)
        np.put(img, (ind[0]) * img.shape[0] + ind[1], y_km + 1)
        return torch.from_numpy(img)

    def adaptive_interval_sampling(self, rays_o, rays_d, z_vals,
                                   N_samples, N_rays, lindisp):
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        x, y = pts[..., 1].to(torch.long), pts[..., 0].to(torch.long)
        batch_indexes = self.heights_img[x, y]
        # print("batch indexs: ", batch_indexes.shape)

        adaptive_z_vals = None

        for ray in batch_indexes:
            ray = ray.cpu().numpy()
            tmp_result = np.where(np.diff(ray) != 0)[0]
            tmp_result = sorted(np.concatenate([tmp_result, tmp_result + 1]))
            # print("tmp_result: ", tmp_result)
            if len(tmp_result) == 0:
                # resampling_near.append([0])
                # resampling_far.append([16])
                N_samples = 64
                near = 0
                far = 16
                t_vals = torch.linspace(0., 1., steps=N_samples)
                if not lindisp:
                    z_vals = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
                z_vals = z_vals.unsqueeze(0)

            elif len(tmp_result) == 1*2:
                # resampling_near.append([0])
                # resampling_far.append([16])
                N_samples = [16, 32, 16]

                near = 0
                far = tmp_result[0]
                t_vals = torch.linspace(0., 1., steps=N_samples[0])
                if not lindisp:
                    z_vals_1 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_1 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[0]
                far = tmp_result[1]
                t_vals = torch.linspace(0., 1., steps=N_samples[1])
                if not lindisp:
                    z_vals_2 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_2 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[1]
                far = 16
                if near >= 16:
                    far = near + 2
                t_vals = torch.linspace(0., 1., steps=N_samples[2])
                if not lindisp:
                    z_vals_3 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_3 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
                z_vals = torch.cat([z_vals_1, z_vals_2, z_vals_3], dim=0)
                z_vals = z_vals.unsqueeze(0)

            elif len(tmp_result) == 2*2:
                # resampling_near.append([tmp_result[0]])
                # resampling_far.append([tmp_result[1]])
                # near = torch.Tensor(resampling_near)
                # far = torch.Tensor(resampling_far)
                N_samples = [9, 18, 10, 18, 9]

                near = 0
                far = tmp_result[0]
                t_vals = torch.linspace(0., 1., steps=N_samples[0])
                if not lindisp:
                    z_vals_1 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_1 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[0]
                far = tmp_result[1]
                t_vals = torch.linspace(0., 1., steps=N_samples[1])
                if not lindisp:
                    z_vals_2 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_2 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[1]
                far = tmp_result[2]
                t_vals = torch.linspace(0., 1., steps=N_samples[2])
                if not lindisp:
                    z_vals_3 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_3 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[2]
                far = tmp_result[3]
                t_vals = torch.linspace(0., 1., steps=N_samples[3])
                if not lindisp:
                    z_vals_4 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_4 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[3]
                far = 16
                if near >= 16:
                    far = near + 2
                t_vals = torch.linspace(0., 1., steps=N_samples[4])
                if not lindisp:
                    z_vals_5 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_5 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                z_vals = torch.cat([z_vals_1, z_vals_2, z_vals_3, z_vals_4, z_vals_5], dim=0)
                z_vals = z_vals.unsqueeze(0)

            elif len(tmp_result) == 3*2:
                N_samples = [6, 14, 6, 13, 6, 13, 6]

                near = 0
                far = tmp_result[0]
                t_vals = torch.linspace(0., 1., steps=N_samples[0])
                if not lindisp:
                    z_vals_1 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_1 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[0]
                far = tmp_result[1]
                t_vals = torch.linspace(0., 1., steps=N_samples[1])
                if not lindisp:
                    z_vals_2 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_2 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[1]
                far = tmp_result[2]
                t_vals = torch.linspace(0., 1., steps=N_samples[2])
                if not lindisp:
                    z_vals_3 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_3 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[2]
                far = tmp_result[3]
                t_vals = torch.linspace(0., 1., steps=N_samples[3])
                if not lindisp:
                    z_vals_4 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_4 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[3]
                far = tmp_result[4]
                t_vals = torch.linspace(0., 1., steps=N_samples[4])
                if not lindisp:
                    z_vals_5 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_5 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[4]
                far = tmp_result[5]
                t_vals = torch.linspace(0., 1., steps=N_samples[5])
                if not lindisp:
                    z_vals_6 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_6 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[5]
                far = 16
                if near >= 16:
                    far = near + 2
                t_vals = torch.linspace(0., 1., steps=N_samples[6])
                if not lindisp:
                    z_vals_7 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_7 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                z_vals = torch.cat([z_vals_1, z_vals_2, z_vals_3, z_vals_4, z_vals_5, z_vals_6, z_vals_7], dim=0)
                z_vals = z_vals.unsqueeze(0)

            elif len(tmp_result) == 4*2:
                N_samples = [5, 10, 5, 10, 5, 10, 5, 10, 4]

                near = 0
                far = tmp_result[0]
                t_vals = torch.linspace(0., 1., steps=N_samples[0])
                if not lindisp:
                    z_vals_1 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_1 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[0]
                far = tmp_result[1]
                t_vals = torch.linspace(0., 1., steps=N_samples[1])
                if not lindisp:
                    z_vals_2 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_2 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[1]
                far = tmp_result[2]
                t_vals = torch.linspace(0., 1., steps=N_samples[2])
                if not lindisp:
                    z_vals_3 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_3 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[2]
                far = tmp_result[3]
                t_vals = torch.linspace(0., 1., steps=N_samples[3])
                if not lindisp:
                    z_vals_4 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_4 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[3]
                far = tmp_result[4]
                t_vals = torch.linspace(0., 1., steps=N_samples[4])
                if not lindisp:
                    z_vals_5 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_5 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[4]
                far = tmp_result[5]
                t_vals = torch.linspace(0., 1., steps=N_samples[5])
                if not lindisp:
                    z_vals_6 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_6 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[5]
                far = tmp_result[6]
                t_vals = torch.linspace(0., 1., steps=N_samples[6])
                if not lindisp:
                    z_vals_7 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_7 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[6]
                far = tmp_result[7]
                t_vals = torch.linspace(0., 1., steps=N_samples[7])
                if not lindisp:
                    z_vals_8 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_8 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[7]
                far = 16
                if near >= 16:
                    far = near + 2
                t_vals = torch.linspace(0., 1., steps=N_samples[8])
                if not lindisp:
                    z_vals_9 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_9 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                z_vals = torch.cat([z_vals_1, z_vals_2, z_vals_3, z_vals_4, z_vals_5, z_vals_6, z_vals_7, z_vals_8, z_vals_9], dim=0)
                z_vals = z_vals.unsqueeze(0)

            elif len(tmp_result) == 5*2:
                N_samples = [4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4]

                near = 0
                far = tmp_result[0]
                t_vals = torch.linspace(0., 1., steps=N_samples[0])
                if not lindisp:
                    z_vals_1 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_1 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[0]
                far = tmp_result[1]
                t_vals = torch.linspace(0., 1., steps=N_samples[1])
                if not lindisp:
                    z_vals_2 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_2 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[1]
                far = tmp_result[2]
                t_vals = torch.linspace(0., 1., steps=N_samples[2])
                if not lindisp:
                    z_vals_3 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_3 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[2]
                far = tmp_result[3]
                t_vals = torch.linspace(0., 1., steps=N_samples[3])
                if not lindisp:
                    z_vals_4 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_4 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[3]
                far = tmp_result[4]
                t_vals = torch.linspace(0., 1., steps=N_samples[4])
                if not lindisp:
                    z_vals_5 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_5 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[4]
                far = tmp_result[5]
                t_vals = torch.linspace(0., 1., steps=N_samples[5])
                if not lindisp:
                    z_vals_6 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_6 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[5]
                far = tmp_result[6]
                t_vals = torch.linspace(0., 1., steps=N_samples[6])
                if not lindisp:
                    z_vals_7 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_7 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[6]
                far = tmp_result[7]
                t_vals = torch.linspace(0., 1., steps=N_samples[7])
                if not lindisp:
                    z_vals_8 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_8 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[7]
                far = tmp_result[8]
                t_vals = torch.linspace(0., 1., steps=N_samples[8])
                if not lindisp:
                    z_vals_9 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_9 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[8]
                far = tmp_result[9]
                t_vals = torch.linspace(0., 1., steps=N_samples[9])
                if not lindisp:
                    z_vals_10 = near * (1.-t_vals) + far * (t_vals)
                else:
                    z_vals_10 = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

                near = tmp_result[9]
                far = 16
                if near >= 16:
                    far = near + 2
                t_vals = torch.linspace(0., 1., steps=N_samples[10])
                if not lindisp:
                    z_vals_11 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_11 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                z_vals = torch.cat([z_vals_1, z_vals_2, z_vals_3, z_vals_4, z_vals_5, z_vals_6, z_vals_7, z_vals_8, z_vals_9, z_vals_10, z_vals_11], 0)
                z_vals = z_vals.unsqueeze(0)

            elif len(tmp_result) == 6 * 2:
                N_samples = [2, 9, 2, 9, 2, 8, 2, 8, 2, 8, 2, 8, 2]

                near = 0
                far = tmp_result[0]
                t_vals = torch.linspace(0., 1., steps=N_samples[0])
                if not lindisp:
                    z_vals_1 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_1 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[0]
                far = tmp_result[1]
                t_vals = torch.linspace(0., 1., steps=N_samples[1])
                if not lindisp:
                    z_vals_2 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_2 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[1]
                far = tmp_result[2]
                t_vals = torch.linspace(0., 1., steps=N_samples[2])
                if not lindisp:
                    z_vals_3 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_3 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[2]
                far = tmp_result[3]
                t_vals = torch.linspace(0., 1., steps=N_samples[3])
                if not lindisp:
                    z_vals_4 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_4 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[3]
                far = tmp_result[4]
                t_vals = torch.linspace(0., 1., steps=N_samples[4])
                if not lindisp:
                    z_vals_5 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_5 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[4]
                far = tmp_result[5]
                t_vals = torch.linspace(0., 1., steps=N_samples[5])
                if not lindisp:
                    z_vals_6 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_6 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[5]
                far = tmp_result[6]
                t_vals = torch.linspace(0., 1., steps=N_samples[6])
                if not lindisp:
                    z_vals_7 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_7 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[6]
                far = tmp_result[7]
                t_vals = torch.linspace(0., 1., steps=N_samples[7])
                if not lindisp:
                    z_vals_8 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_8 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[7]
                far = tmp_result[8]
                t_vals = torch.linspace(0., 1., steps=N_samples[8])
                if not lindisp:
                    z_vals_9 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_9 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[8]
                far = tmp_result[9]
                t_vals = torch.linspace(0., 1., steps=N_samples[9])
                if not lindisp:
                    z_vals_10 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_10 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[9]
                far = tmp_result[10]
                t_vals = torch.linspace(0., 1., steps=N_samples[10])
                if not lindisp:
                    z_vals_11 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_11 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[10]
                far = tmp_result[11]
                t_vals = torch.linspace(0., 1., steps=N_samples[11])
                if not lindisp:
                    z_vals_12 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_12 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                near = tmp_result[11]
                far = 16
                if near >= 16:
                    far = near + 2
                t_vals = torch.linspace(0., 1., steps=N_samples[12])
                if not lindisp:
                    z_vals_13 = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals_13 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                z_vals = torch.cat([z_vals_1, z_vals_2, z_vals_3, z_vals_4, z_vals_5, z_vals_6, z_vals_7, z_vals_8, z_vals_9, z_vals_10, z_vals_11, z_vals_12, z_vals_13], dim=0)
                z_vals = z_vals.unsqueeze(0)

            else:
                print(f'Invalid number of resampling points {len(tmp_result)}')

            # z_vals = z_vals.unsqueeze(0)
            if adaptive_z_vals is None:
                adaptive_z_vals = z_vals
            else:
                # print('z_vals_concated', z_vals_concated.shape)
                adaptive_z_vals = torch.cat([adaptive_z_vals, z_vals], dim=0)
        # print("length of tmp_result is ", len(tmp_result))
        # print("train is ", train)
        N_samples = 64
        z_vals = adaptive_z_vals.expand([N_rays, N_samples])

        return z_vals
