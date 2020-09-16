import jax.numpy as np
from jax import grad, nn, random, jit
from jax.experimental import stax, optimizers
from jax.experimental.optimizers import l2_norm, clip_grads
from jax.numpy import linalg
from tqdm import tqdm


def predict(W, X):
  """Forward propagation for logistic regression."""
  return nn.sigmoid(np.dot(X, W))

def loss(W, X, y, l2=0.):
  """Binary cross entropy loss with l2 regularization."""
  y_hat = predict(W, X)
  bce = y * np.log(y_hat) + (1. - y) * np.log(1. - y_hat)
  return -np.mean(bce) + l2 * l2_norm(W)

def unit_projection(W):
  """Projects model parameters to have at most l2 norm of 1."""
  return clip_grads(W, 1)

def step(W, X, y, l2=0., proj=unit_projection):
  """A single step of projected gradient descent."""
  g = grad(loss)(W, X, y, l2)
  W = W - 0.5 * g
  W = proj(W)
  return W

def train(W, X, y, l2=0., iters=1):
  """Simply executes several model parameter steps."""
  for i in range(iters):
    W = step(W, X, y, l2)
  return W

def process_update(W, X, y, update, train):
  """
  Updates the dataset according to some update function (e.g. append datum, delete datum) then
  finetunes the model on the resulting dataset according to some given training function.
  """
  X, y = update(X, y)
  W = train(W, X, y)
  return W, X, y

def process_updates(W, X, y, updates, train):
  """Processes a sequence of updates."""
  for update in updates:
    W, X, y = process_update(W, X, y, update, train)
  return W, X, y

def compute_sigma(num_examples, iterations, lipshitz, strong, epsilon, delta):
  """Theorem 3.1 https://arxiv.org/pdf/2007.02923.pdf"""
  gamma = (smooth - strong) / (smooth + strong)
  numerator = 4 * np.sqrt(2) * lipshitz * np.power(gamma, iterations)
  denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * ((np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta)))
  return numerator / denominator

def publish(rng, W, sigma):
  """Publishing function which adds Gaussian noise with scale sigma."""
  return W + sigma * random.normal(rng, W.shape)

def accuracy(W, X, y):
  """Computes the model accuracy given a dataset."""
  y_hat = (predict(W, X) > 0.5).astype(np.int32)
  return np.mean(y_hat == y)

def delete_index(idx, *args):
  """Deletes index `idx` from each of args (assumes they all have same shape)."""
  mask = np.eye(len(args[0]))[idx] == 0.
  return (arg[mask] for arg in args)

def append_datum(data, *args):
  return (np.concatenate((arg, datum)) for arg, datum in zip(args, data))

if __name__ == "__main__":
  rng = random.PRNGKey(0)

  num_train = 1000
  num_test = 200
  num_updates = 25

  init_iterations = 1000
  update_iterations = 25

  l2 = 0.05
  strong = l2
  smooth = 4 - l2
  diameter = 2
  lipshitz = 1 + l2

  epsilon = 5
  delta = 1 / (num_train ** 2)

  # Two dimensional Gaussian points with label 1 if above Y = 0 and 0 otherwise
  X = random.normal(rng, shape=(num_train, 2)) # (num_train, 2)
  y = (X[:, 0] > 0.).astype(np.int32) # (num_train,)

  X_test = random.normal(rng, shape=(num_test, 2)) # (num_test, 2)
  y_test = (X_test[:, 0] > 0.).astype(np.int32) # (num_test,)

  W = np.ones((X.shape[1],)) # (2,)
  W = unit_projection(W)
  W = train(W, X, y, l2, init_iterations)

  # Delete first row `num_updates` times in sequence
  updates = [lambda X, y: delete_index(0, X, y) for i in range(num_updates)]
  train_fn = lambda W, X, y: train(W, X, y, l2, update_iterations)

  print('Processing updates...')
  W, X, y = process_updates(W, X, y, updates, train_fn)
  print('Accuracy: {:.4f}\n'.format(accuracy(W, X_test, y_test)))

  sigma = compute_sigma(num_train, update_iterations, lipshitz, strong, epsilon, delta)
  print('Epsilon: {}, Delta: {}, Sigma: {:.4f}'.format(epsilon, delta, sigma))

  temp, rng = random.split(rng)
  W = publish(temp, W, sigma)
  print('Accuracy (published): {:.4f}'.format(accuracy(W, X_test, y_test)))
