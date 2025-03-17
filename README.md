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

self.counter = 0
self.cycler = ParameterCycler(parameters=[self.tscale, self.matrix])  # <-- add learnables

self.cycler.toggle_requires_grad()  # <-- goes in forward

# example
self.tscale = nn.Parameter(torch.ones(1), requires_grad=False) 
self.matrix = nn.Parameter(torch.eye(self.head_dim), requires_grad=False)




```
