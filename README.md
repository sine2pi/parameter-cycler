maybe it can be used for something..
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

```
