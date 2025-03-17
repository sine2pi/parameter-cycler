maybe it cam be used for something..
Random or not so random parameter cycler.. 
it turns learnable parameters on and off randomly every forwad pass.. hundred a minute. Spread the learnable load .. as they say.

```python

class ParameterCycler:
    def __init__(self, parameters):
        self.parameters = parameters
        self.current_idx = 0
    def toggle_requires_grad(self):
        x = random.randint(0, len(self.parameters) - 1)
        for x, param in enumerate(self.parameters):
            param.requires_grad = (x == self.current_idx)
            print(f"Parameter {x}: requires_grad={param.requires_grad}")
        self.current_idx = (self.current_idx + 1) % len(self.parameters)

class rotary(nn.Module):
    def __init__(self, dims, head, freq=10000):
        super().__init__()
        self.dims = dims
        self.head = head
        self.freq = freq
        self.head_dim = self.dims // self.head
        self.rot = self.head_dim // 2
        self.counter = 0
        
        self.thetas = nn.Parameter(torch.zeros(self.rot))
        self.pairs = nn.Parameter(torch.rand(self.rot, 2) * self.head_dim)
        self.tscale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rscale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.matrix = nn.Parameter(torch.eye(self.head_dim), requires_grad=False)
        
        self.freq_data = 1.0 / (self.freq ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.invf = nn.Parameter(self.freq_data, requires_grad=False)

        self.cycler = ParameterCycler(parameters=[self.thetas, self.tscale, self.rscale, self.matrix, self.invf])
        self.reset_parameters()
        
    def update_base(self, freq):
        if freq != self.freq:
            self.freq = freq
            self.invf.data.copy_(1.0 / (self.freq ** (torch.arange(start=0, end=self.head_dim, step=2).float() / self.head_dim)))
            self.update_pairs()    
        
    def update_pairs(self):
        self.p = []
        while len(self.p) < self.rot:
            i, j = torch.randint(low=0, high=self.head_dim - 1, size=(2,))
            if i != j and (i, j) not in self.p and (j, i) not in self.p:
                self.p.append((i, j))
        self.p.data.copy_(src=torch.tensor(data=self.p, dtype=torch.float32))
        
    def reset_parameters(self):
        nn.init.orthogonal_(self.matrix)
        nn.init.zeros_(self.thetas)

    def q_rotation(self, x, theta, u, v):
        u = u / torch.norm(u)
        v = v / torch.norm(v)

        half_theta = theta / 2
        cos_ht = torch.cos(half_theta)
        sin_ht = torch.sin(half_theta)

        q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
        q_conj = torch.cat([cos_ht.unsqueeze(0), -sin_ht * u])

        x_shape = x.shape
        x = x.view(-1, 3)

        uv_cross = torch.cross(u.unsqueeze(0), x)
        uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        x_rot = x + 2 * (q[0] * uv_cross + uuv_cross)

        x_rot = x_rot.view(*x_shape)
        return x_rot

    def rotation_matrix(self, dims, i, j, theta):
        G = torch.eye(dims, device=theta.device)
        c, s = torch.cos(theta), torch.sin(theta)
        G[i, i], G[j, j] = c, c
        G[i, j], G[j, i] = -s, s

        if dims == 3:
            u = torch.eye(dims, device=theta.device)[i]
            v = torch.eye(dims, device=theta.device)[j]
            Q = self.q_rotation(
                torch.eye(dims, device=theta.device), theta=theta, u=u, v=v)
            G = (G + Q) / 2
        return G

    def apply_rotations(self, x):
        adjust = int(torch.round(self.rscale * self.rot))
        for k in range(adjust):
            i, j = self.pairs[k].long()
            theta = self.thetas[k] * self.tscale
            G = self.rotation_matrix(self.head_dim, i.item(), j.item(), theta)
            x = x @ G
        return x

    def forward(self, x):
        batch, ctx, *rest = x.size()

        if len(rest) == 1:
            self.dims = rest[0]
            if self.dims != self.head * self.head_dim:
                raise ValueError(
                    f"Needed {self.head * self.head_dim}, but got too many {dims}"
                )
        elif len(rest) == 2:
            self.head, self.head_dim = rest
            if self.head != self.head or self.head_dim != self.head_dim:
                raise ValueError(
                    f"This many head {self.head} and head_dims {self.head_dim} we need, got this many head {head} and head_dims {head_dim} we did."
                )
        else:
            raise ValueError(f"Expected the thingy to be 3D or 4D, but got {x.dim()}D")

        self.cycler.toggle_requires_grad()

        x = x.view(batch, ctx, self.head, self.head_dim)
        x = x.reshape(-1, self.head_dim)

        x = self.apply_rotations(x)
        x = x @ self.matrix

        x = x.view(batch, ctx, self.head, self.head_dim)

        position = torch.arange(ctx, device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = self.invf.unsqueeze(0)
        sinusoid_inp = position * div_term

        sin = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(2)
        cos = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(2)

        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x = x.view(batch, ctx, self.dims)
        x = x * math.sqrt(self.dims)
        return x

# def normalize(v, eps=1e-8):
#     return v / (torch.norm(v) + eps)

# class Rotary(nn.Module):
#     def __init__(s, dims, head, theta=10000, tscale=None, rscale=None, device=None):
#         super().__init__()
#         device = device or torch.device('cpu')
#         head_dim = dims // head
#         rot = head_dim // 2

#         tscale = nn.Parameter(data=torch.ones(1, device=device), requires_grad=True)
#         rscale = nn.Parameter(data=torch.ones(1, device=device), requires_grad=True)
#         thetas = nn.Parameter(data=torch.zeros(rot, device=device))
#         pairs = nn.Parameter(data=torch.rand(rot, 2, device=device) * head_dim)
#         matrix = nn.Parameter(data=torch.eye(head_dim, device=device), requires_grad=True)
        
#         scale_factor = tscale ** (dims / max(dims - 2, 1))  # Safe scaling
#         theta *= scale_factor
#         findices = torch.arange(0, head_dim, 2, device=device).float()
#         fdata = 1.0 / (theta ** (findices / head_dim))
#         invf = nn.Parameter(data=fdata, requires_grad=True)

#         s.register_buffer("cached_freqs", torch.zeros_like(fdata, device=device), persistent=False)
#         s.dims, s.head = dims, head
#         s.matrix, s.invf, s.tscale, s.rscale, s.thetas, s.pairs = tscale, rscale, thetas, pairs, matrix, invf
#         counter = 0
#         cycler = ParameterCycler(parameters=[thetas, pairs, tscale, rscale, matrix, invf])
        
#         s.max_ctx = 0
#         s.cache = 0
#         s.freqs = fdata
#         s.device = device
                
#     # def q_rotation(s, x, theta, u, v):
#     #     u, v = normalize(u), normalize(v)
#     #     half_theta = theta / 2
#     #     cos_ht, sin_ht = torch.cos(half_theta), torch.sin(half_theta)
#     #     q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
#     #     uv_cross = torch.cross(u.unsqueeze(0), x)
#     #     uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
#     #     return x + 2 * (q[0] * uv_cross + uuv_cross)
        
#     def reset_parameters(s, matrix, thetas):
#         with torch.no_grad():
#             nn.init.orthogonal_(matrix)
#             nn.init.zeros_(thetas)


#     def normalize(s, v, eps=1e-8):
#         return v / (torch.norm(v) + eps)

#     def reset_parameters(self):
#         nn.init.orthogonal_(tensor=self.matrix)
#         nn.init.zeros_(tensor=self.thetas)

#     def q_rotation(s, x, theta, u, v):
#         x = x.to(s.device, s.dtype)
#         theta = theta.to(s.device, s.dtype) if not isinstance(theta, (int, float)) else theta
#         u = u.to(s.device)
#         v = v.to(s.device)
        
#         u = u / torch.norm(u)
#         v = v / torch.norm(v)

#         half_theta = theta / 2
#         cos_ht = torch.cos(half_theta)
#         sin_ht = torch.sin(half_theta)

#         q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
#         q_conj = torch.cat([cos_ht.unsqueeze(0), -sin_ht * u])

#         x_shape = x.shape
#         x = x.view(-1, 3)

#         uv_cross = torch.cross(u.unsqueeze(0), x)
#         uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
#         x_rot = x + 2 * (q[0] * uv_cross + uuv_cross)

#         x_rot = x_rot.view(*x_shape)
#         return x_rot

#     def rotation_matrix(s, dims, i, j, theta):
#         G = torch.eye(dims, device=self.device)
#         c, s = torch.cos(theta), torch.sin(theta)
#         G[i, i], G[j, j] = c, c
#         G[i, j], G[j, i] = -s, s

#         if dims == 3:
#             u = torch.eye(dims, device=self.device)[i]
#             v = torch.eye(dims, device=self.device)[j]
#             Q = q_rotation(
#                 torch.eye(dims, device=self.device), theta=theta, u=u, v=v)
#             G = (G + Q) / 2
#         return G

#     def rotate(s, x):
#         rotate = int(torch.round(s.rscale * s.rot))
#         for k in range(rotate):
#             i, j = s.r_pairs[k].long()
#             theta = s.thetas[k] * s.tscale
#             G = s.rotation_matrix(dims=self.head_dim, i=i.item(), j=j.item(), theta=theta)
#             x = x @ G
#         return x

    
#     def forward(s, x):
#         bat, ctx, *rest = x.size()

#         if len(rest) == 1:
#             dims = rest[0]
#             if dims != s.head * s.head_dim:
#                 raise ValueError(
#                     f"Needed {s.head * s.head_dim}, but got too many {dims}"
#                 )
#         elif len(rest) == 2:
#             head, head_dim = rest
#             if head != s.head or head_dim != s.head_dim:
#                 raise ValueError(f"This many head {s.head} and head_dims {s.head_dim} we need, got this many head {head} and head_dims {head_dim} we did."                )
#         else:
#             raise ValueError(f"Expected the thingy to be 3D or 4D, but got {x.dim()}D")
        
#         if s.cache > 0:
#             if ctx > s.max_ctx:
#                 s.max_ctx = ctx
#                 s.cached_freqs = s.freqs.repeat(s.cache, 1)
                
#         for param in s.parameters:
#             param.requires_grad = False
#         s.parameters[s.counter].requires_grad = True
#         s.counter = (s.counter + 1) % len(s.parameters)

#         x = x.view(bat, ctx, s.head, s.head_dim)
#         x = x.reshape(-1, s.head_dim)

#         x = s.rotate(x)
#         x = x @ s.matrix

#         x = x.view(bat, ctx, s.head, s.head_dim)

#         position = torch.arange(ctx, device=x.device, dtype=x.dtype).unsqueeze(1)
#         div_term = s.inv_freq.unsqueeze(0)
#         sinusoid_inp = position * div_term

#         sin = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(2)
#         cos = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(2)

#         x1, x2 = x[..., ::2], x[..., 1::2]
#         x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
#         x = x.view(bat, ctx, s.dims)
#         x = x * math.sqrt(s.dims)
#         return x


```
