import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

# Generate fixed dataset
key = jr.PRNGKey(0)
N = 10000
x = jr.uniform(key, (N,), minval=-2, maxval=2)
c = jr.choice(key, jnp.array([0, 1]), shape=(N,))
y = jnp.sin(x) + 2 * c
data = jnp.stack([x, y], axis=1)


# Simple MLP score model
class ScoreModel(eqx.Module):
    mlp: eqx.nn.MLP

    def __call__(self, x, t):
        def single(x_, t_):
            inp = jnp.concatenate([x_, t_], axis=0)  # (3,)
            return self.mlp(inp)

        return jax.vmap(single)(x, t)


model_key, train_key = jr.split(key)
model = ScoreModel(eqx.nn.MLP(3, 2, 32, 4, key=model_key))
opt = optax.adam(1e-4)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

batch_size = 512
steps = 20000
plot_every = 4000


@eqx.filter_jit
def loss_fn(model, key):
    dkey, tkey, epskey = jr.split(key, 3)
    t = jr.uniform(tkey, (batch_size, 1), minval=1e-5, maxval=1.0)
    idx = jr.randint(dkey, (batch_size,), 0, data.shape[0])
    x0 = data[idx]
    eps = jr.normal(epskey, (batch_size, 2))
    xt = x0 + eps * jnp.sqrt(t)
    score_pred = model(xt, t)
    return jnp.mean((score_pred * jnp.sqrt(t) + eps) ** 2)


@eqx.filter_jit
def train_step(model, opt_state, key):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, key)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss.mean()


def sample_reverse_diffusion(model, key, n=2000, steps=50):
    dt = 1.0 / steps
    xs = jr.normal(key, (n, 2))
    for i in reversed(range(1, steps + 1)):
        t = i * dt
        tt = jnp.full((n, 1), t)
        score = model(xs, tt)
        z = jr.normal(key, (n, 2))
        xs = xs + score * dt
    return xs


fig, axs = plt.subplots(1, (steps // plot_every) + 1, figsize=(15, 3))
for i in range(steps + 1):
    model, opt_state, loss = train_step(model, opt_state, jr.fold_in(train_key, i))
    if i % plot_every == 0:
        samples = sample_reverse_diffusion(model, jr.fold_in(train_key, i))
        axs[i // plot_every].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
        axs[i // plot_every].set_title(f"Step {i}")

    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss}")
plt.tight_layout()

plt.figure(figsize=(5, 5))
plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
plt.title("Original Data Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
