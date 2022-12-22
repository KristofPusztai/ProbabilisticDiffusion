# ProbabilisticDiffusion
This is a PyTorch implementation of the training algorithm found in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

Specifically, we implement the following training procedure:
![image](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM-algo.png)

Where $\epsilon_\theta$  represents the user defined model with learnable parameters $\theta$.
## Usage
### Defining Model
#### Defining $\epsilon_\theta$
Below we define our model with the parameters we would like to learn. In this case we use a simple
architecture with a combination of PyTorch Linear layers with Softplus activations, as well as an embedding to take into
account the timestep, $t$, which we also include in the input (the y value).
```
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out
class ConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = nn.Linear(128, 2)
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)
```
#### Defining Diffusiong Based Learning Model
We define our diffusion based model with 200 timesteps, MSE loss (although the original algorithm specifies just SSE but we found that MSE works as well),
 beta start and end values of 1e-5, 1e-2 respectively with a linear schedule, and use the 
ADAM optimizer with a learning rate of 1e-3.
```
n_steps=200
model = ConditionalModel(n_steps)
loss = torch.nn.MSELoss(reduction='mean') # We use MSE for the loss which adheres to the gradient step procedure defined
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # ADAM Optimizer Parameters for learning
diffusion = Diffusion(data, n_steps, 1e-5, 1e-2, 'linear', model, loss, optimizer) # Note the (1e-5, 1e-2) are Beta start and end values
```
#### Forward Sampling
This allows us to see the forward diffusion process and ensure that
our n_steps parameter is large enough. We want to see the data morph into
standard gaussian distributed points by the last time step.
```
noised = diffusion.forward(199, s=5)
plt.show()
stats.probplot(noised[:,0], dist="norm", plot=plt)
plt.show()
```
TODO: Put in plots 
### Training
We train with batch size of 1,000 for 10,000 epochs.
```
diffusion.train(1000, 10000)
```
### Sampling New Data
We can sample new data based on the learned model via the following method:
```
new_x = diffusion.sample(1000, 50, s=3)
```
TODO: Plots

This method automatically plots, but this can be turned off, in addition,
we can specify which points to keep from these new samples, 'last' will only keep
the last timestep of samples, 'all', will keep all timesteps, and for more
granulatiry, one can specify a tuple of integer values corresponding
to the desired timesteps to keep.