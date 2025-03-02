import jax
import jax.numpy as jnp
from jax import random, jit
import optax
import numpy as np
from utils import load_images_from_folder, load_camera_parameters, positional_encoding

class NeRFModel:
    def __init__(self, rng_key, input_dim, hidden_dim, num_layers, output_dim):
        self.params = self.init_params(rng_key, input_dim, hidden_dim, num_layers, output_dim)
        
    def init_params(self, key, in_dim, hid_dim, layers, out_dim):
        params = []
        keys = random.split(key, layers+1)
        params.append({'W': random.normal(keys[0], (in_dim, hid_dim)) * 0.1,
                       'b': jnp.zeros((hid_dim,))})
        for i in range(1, layers-1):
            params.append({'W': random.normal(keys[i], (hid_dim, hid_dim)) * 0.1,
                           'b': jnp.zeros((hid_dim,))})
        params.append({'W': random.normal(keys[-1], (hid_dim, out_dim)) * 0.1,
                       'b': jnp.zeros((out_dim,))})
        return params

    def __call__(self, params, x):
        h = x
        for layer in params[:-1]:
            h = jnp.dot(h, layer['W']) + layer['b']
            h = jax.nn.relu(h)
        out = jnp.dot(h, params[-1]['W']) + params[-1]['b']
        return out

@jit
def loss_fn(params, model, batch):
    inputs, targets = batch
    preds = model(params, inputs)
    return jnp.mean((preds - targets) ** 2)

def train_teacher_model(image_folder, camera_file, num_epochs=1000, lr=1e-3):
    images = load_images_from_folder(image_folder)
    cam_params = load_camera_parameters(camera_file)
    # For demonstration, we assume dummy flattened inputs and targets.
    rng = random.PRNGKey(0)
    model = NeRFModel(rng, input_dim=63, hidden_dim=256, num_layers=10, output_dim=4*16)  # density+SH coefficients
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(model.params)

    @jit
    def update(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, model, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    for epoch in range(num_epochs):
        # Create a dummy batch (replace with actual data loader)
        inputs = random.normal(rng, (32, 63))
        targets = random.normal(rng, (32, 4*16))
        batch = (inputs, targets)
        model.params, opt_state, loss = update(model.params, opt_state, batch)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    return model

if __name__ == '__main__':
    teacher = train_teacher_model("data/images", "data/cam_params.npy")
