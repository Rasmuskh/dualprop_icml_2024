import jax
from jax import vmap # for parallelizing operations accross multiple datapoints (and managing rng across them)
import jax.numpy as jnp # JAX NumPy
from jax.tree_util import tree_map
from flax.training import train_state # Useful dataclass to keep train state
from flax import traverse_util
from flax.core.frozen_dict import unfreeze
import dm_pix as pix # for data augmentation

import numpy as np # regular old numpy
import optax # Optimizers
import tensorflow as tf
import tensorflow_datasets as tfds # TFDS for loading datasets
import time, datetime # measuring runetime and generating timestamps for experiments
import os # for os.makdirs() function
import seaborn as sns
import matplotlib.pylab as plt

class SumGreaterThan100Error(Exception):
    pass

def compute_metrics(image, labels_onehot, labels, state):
    
    s = state.apply_fn({'params':  state.params}, image, method='ff_with_hiddens')
    logits = s[-1]

    def loss_fn():
        loss = state.apply_fn({'params':  state.params}, logits, labels_onehot, method='output_loss')
        return loss
    loss = loss_fn()/logits.shape[0]
    accuracy = 100.0*jnp.mean(jnp.argmax(logits, -1) == labels, dtype=jnp.float32)

    _, top5_indices = jax.lax.top_k(logits, 5) 
    top5accuracy = 100.0*((top5_indices == labels[:,None]).sum())/logits.shape[0]

    metrics = {'loss': loss, 
               'accuracy': accuracy, 
               'top5accuracy':top5accuracy,
               }
    return metrics

def get_mnist(dtype, percent_train, percent_val):
    """Load train, test and val datasets into memory."""
    if percent_val + percent_train > 100:
        raise SumGreaterThan100Error("sum of percent_train and percent_val should be less than 100.")
    
    MEAN = 0.1307
    STD = 0.3081
    val_start = 100 - percent_val
    ds_builder = tfds.builder('mnist', data_dir='data')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[:{percent_train}%]', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[{val_start}%:]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    train_ds['image'] = ((dtype(train_ds['image']) / 255.) - MEAN)/STD
    val_ds['image'] = ((dtype(val_ds['image']) / 255.) - MEAN)/STD
    test_ds['image'] = ((dtype(test_ds['image']) / 255.) - MEAN)/STD

    train_ds['image'] = jnp.pad(train_ds['image'], ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=((0,0), (0,0), (0, 0), (0,0)))
    val_ds['image'] = jnp.pad(val_ds['image'], ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=((0,0), (0,0), (0, 0), (0,0)))
    test_ds['image'] = jnp.pad(test_ds['image'], ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=((0,0), (0,0), (0, 0), (0,0)))

    return train_ds, val_ds, test_ds

def get_svhn(dtype, percent_train, percent_val):
    """Load train, test and val datasets into memory."""
    if percent_val + percent_train > 100:
        raise SumGreaterThan100Error("sum of percent_train and percent_val should be less than 100.")
    
    val_start = 100 - percent_val
    ds_builder = tfds.builder('svhn_cropped', data_dir='data')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[:{percent_train}%]', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[{val_start}%:]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    train_ds['image'] = dtype(train_ds['image']) / 255.
    val_ds['image'] = dtype(val_ds['image']) / 255.
    test_ds['image'] = dtype(test_ds['image']) / 255.

    mean_data = jnp.expand_dims(jnp.array([0.4377, 0.4438, 0.4728]), axis=(0,1))
    std_data = jnp.expand_dims(jnp.array([0.1980, 0.2010, 0.1970]), axis=(0,1))
    train_ds['image'] = (train_ds['image'] - mean_data)/std_data
    val_ds['image'] = (val_ds['image'] - mean_data)/std_data
    test_ds['image'] = (test_ds['image'] - mean_data)/std_data

    return train_ds, val_ds, test_ds

def get_fashionmnist(dtype, percent_train, percent_val):
    """Load train, test and val datasets into memory."""
    if percent_val + percent_train > 100:
        raise SumGreaterThan100Error("sum of percent_train and percent_val should be less than 100.")
    
    MEAN = 0.2860
    STD = 0.3530
    val_start = 100 - percent_val

    ds_builder = tfds.builder('fashion_mnist', data_dir='data')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[:{percent_train}%]', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[{val_start}%:]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    train_ds['image'] = ((dtype(train_ds['image']) / 255.) - MEAN)/STD
    val_ds['image'] = ((dtype(val_ds['image']) / 255.) - MEAN)/STD
    test_ds['image'] = ((dtype(test_ds['image']) / 255.) - MEAN)/STD

    train_ds['image'] = jnp.pad(train_ds['image'], ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=((0,0), (0,0), (0, 0), (0,0)))
    val_ds['image'] = jnp.pad(val_ds['image'], ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=((0,0), (0,0), (0, 0), (0,0)))
    test_ds['image'] = jnp.pad(test_ds['image'], ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=((0,0), (0,0), (0, 0), (0,0)))

    return train_ds, val_ds, test_ds

def get_cifar10(dtype, percent_train, percent_val):
    """Load train, test and val datasets into memory."""
    if percent_val + percent_train > 100:
        raise SumGreaterThan100Error("sum of percent_train and percent_val should be less than 100.")
    
    val_start = 100 - percent_val

    ds_builder = tfds.builder('cifar10', data_dir='data')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[:{percent_train}%]', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[{val_start}%:]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = dtype(train_ds['image']) / 255.
    val_ds['image'] = dtype(val_ds['image']) / 255.
    test_ds['image'] = dtype(test_ds['image']) / 255.
    # Throw away some info we don't need
    train_ds.pop('id', None)
    val_ds.pop('id', None)
    test_ds.pop('id', None)

    mean_data = jnp.expand_dims(jnp.array([0.4914, 0.4822, 0.4465]), axis=(0,1))
    std_data = jnp.expand_dims(jnp.array([0.2023, 0.1994, 0.2010]), axis=(0,1))
    train_ds['image'] = (train_ds['image'] - mean_data)/std_data
    val_ds['image'] = (val_ds['image'] - mean_data)/std_data
    test_ds['image'] = (test_ds['image'] - mean_data)/std_data

    return train_ds, val_ds, test_ds

def get_cifar100(dtype, percent_train, percent_val):
    if percent_val + percent_train > 100:
        raise SumGreaterThan100Error("sum of percent_train and percent_val should be less than 100.")

    val_start = 100 - percent_val

    """Load train, test and val datasets into memory."""
    ds_builder = tfds.builder('cifar100', data_dir='data')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[:{percent_train}%]', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[{val_start}%:]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    train_ds['image'] = dtype(train_ds['image']) / 255.
    val_ds['image'] = dtype(val_ds['image']) / 255.
    test_ds['image'] = dtype(test_ds['image']) / 255.
    # throw away some info we don't need
    train_ds.pop('id', None)
    val_ds.pop('id', None)
    test_ds.pop('id', None)
    train_ds.pop('coarse_label', None)
    val_ds.pop('coarse_label', None)
    test_ds.pop('coarse_label', None)

    mean_data = jnp.expand_dims(jnp.array([0.5074,0.4867,0.4411]), axis=(0,1))
    std_data = jnp.expand_dims(jnp.array([0.2011,0.1987,0.2025]), axis=(0,1))
    train_ds['image'] = (train_ds['image'] - mean_data)/std_data
    val_ds['image'] = (val_ds['image'] - mean_data)/std_data
    test_ds['image'] = (test_ds['image'] - mean_data)/std_data

    return train_ds, val_ds, test_ds

def get_imagenet_32x32(dtype, percent_train=95, percent_val=5):
    """Load data."""
    if percent_val + percent_train > 100:
        raise SumGreaterThan100Error("sum of percent_train and percent_val should be less than 100.")
    
    val_start = 100 - percent_val

    ds_builder = tfds.builder('imagenet_resized/32x32')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[:{percent_train}%]', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split=f'train[{val_start}%:]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='validation', batch_size=-1))

    train_ds['image'] = dtype(train_ds['image']) / 255.
    val_ds['image'] = dtype(val_ds['image']) / 255.
    test_ds['image'] = dtype(test_ds['image']) / 255.
    
    mean_data = jnp.expand_dims(jnp.array([0.485, 0.456, 0.406]), axis=(0,1))
    std_data = jnp.expand_dims(jnp.array([0.229, 0.224, 0.225]), axis=(0,1))
    train_ds['image'] = (train_ds['image'] - mean_data)/std_data
    val_ds['image'] = (val_ds['image'] - mean_data)/std_data
    test_ds['image'] = (test_ds['image'] - mean_data)/std_data
    return train_ds, val_ds, test_ds

def create_train_state(rng, model, image_dims, lr, wlr, lrf, momentum, weight_decay, num_epochs, warmup_epochs, decay_epochs, steps_per_epoch):
    """Creates initial `TrainState`."""
    w, h, ch = image_dims
    x = jnp.ones([1, w, h, ch])
    params = model.init(rng, x)['params']

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=wlr, 
        peak_value=lr, 
        warmup_steps=warmup_epochs*steps_per_epoch, 
        decay_steps = (decay_epochs)*steps_per_epoch, 
        end_value=lrf)

    tx = optax.chain( 
        optax.add_decayed_weights(weight_decay=weight_decay, mask=None),
        optax.sgd(schedule, momentum=momentum),
        )

    return train_state.TrainState.create(apply_fn=model.apply, params=unfreeze(params), tx=tx)

def augment_train(image, batch_rng):
    w, h, c = image.shape

    # Random crop
    image = jnp.pad(image, ((4,4), (4,4), (0,0)), 'constant', constant_values=((0,0), (0, 0), (0,0)))
    image = pix.random_crop(batch_rng, image, (w, h, c))
    # Horizonthal flip
    image = pix.random_flip_left_right(batch_rng, image, probability=0.5)

    return image

vmap_augment_train = vmap(augment_train, in_axes=(0, 0))

def no_aug(image, batch_rng):
    return image



def to_float16(ptree):
    return tree_map(lambda x: x.astype(jnp.float16), ptree)

def to_float32(ptree):
    return tree_map(lambda x: x.astype(jnp.float32), ptree)

def train_epoch(state, train_ds, batch_size, rng, augmentation_on, learning_algorithm, num_classes):
    """Train for a single epoch."""
    t0 = time.time()
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []

    for perm in perms:
        # batch = {k: v[perm, ...] for k, v in train_ds.items()}
        image, labels = train_ds["image"][perm], train_ds["label"][perm]
        labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
        rng, inf_rng, batch_rng = jax.random.split(rng, 3)

        batch_rng = jax.random.split(batch_rng, image.shape[0])
        # image = vmap_augment_train_imagenet(image, batch_rng)

        if learning_algorithm != "backprop":
            state, metrics = train_step(state, image, labels_onehot, labels, batch_rng, inf_rng, augmentation_on)
        elif learning_algorithm == "backprop":
            state, metrics = train_step_bp(state, image, labels_onehot, labels, batch_rng, inf_rng, augmentation_on)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {}
    for key in batch_metrics_np[0].keys():
        if key == "cosine_sim":
            epoch_metrics_np[key] = [m[key] for m in batch_metrics_np]
        else:
            epoch_metrics_np[key] = np.mean([m[key] for m in batch_metrics_np], axis=0)
    runtime = time.time() - t0

    return state, epoch_metrics_np, runtime

@jax.jit
def train_step(state, image, labels_onehot, labels, batch_rng, inf_rng, augmentation_on):
    """Train for a single step."""
    # batch_rng = jax.random.split(batch_rng, batch['image'].shape[0])
    
    # image should have dims batchsize, w, h, ch.
    image = jax.lax.cond(augmentation_on, vmap_augment_train, no_aug, image, batch_rng)
    
    splus, sminus = state.apply_fn({'params': state.params}, image, labels_onehot, inf_rng, method='infer_states_train')

    def loss_fn(params, splus, sminus):
        loss = state.apply_fn({'params': params}, splus, sminus, method='get_J')
        return loss
    # compute logits, gradients and loss and accuracy.
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=False)
    ls, grads = grad_fn(state.params, splus, sminus)

    inf_rng, _ = jax.random.split(inf_rng)
    metrics = compute_metrics(image=image, labels_onehot=labels_onehot, labels=labels, state=state)

    get_ref_grad_angle = True
    metrics = jax.lax.cond(get_ref_grad_angle, ref_grad_and_angle, no_ref_grad_and_angle, state, grads, image, labels_onehot, metrics)

    # The optimizer may modify grads, so we need to compare grads and ref_grads before performing the gradient step.
    state = state.apply_gradients(grads=grads)

    return state, metrics

@jax.jit
def train_step_bp(state, image, labels_onehot, labels, batch_rng, inf_rng, augmentation_on):
    """Train for a single step."""
    # batch_rng = jax.random.split(batch_rng, batch['image'].shape[0])
    
    # image should have dims batchsize, w, h, ch.
    image = jax.lax.cond(augmentation_on, vmap_augment_train, no_aug, image, batch_rng)

    # Compute a reference backprop gradient (but don't use it).
    def loss_fn_ref(params, state):
        logits = state.apply_fn({'params': params}, image)
        loss = state.apply_fn({'params': params}, logits, labels_onehot, method='output_loss')/image.shape[0]
        return loss
    
    # compute logits, gradients and loss and accuracy.
    grad_fn_ref = jax.value_and_grad(loss_fn_ref, argnums=0, has_aux=False)
    
    _, bp_grads = grad_fn_ref(state.params, state)

    sstar = state.apply_fn({'params': state.params}, image, method='make_predictions')
    inf_rng, _ = jax.random.split(inf_rng)
    metrics = compute_metrics(image=image, labels_onehot=labels_onehot, labels=labels, state=state)

    # The optimizer may modify grads, so we need to compare grads and ref_grads before performing the gradient step.
    state = state.apply_gradients(grads=bp_grads)

    return state, metrics

@jax.jit
def eval_step(state, params, image, labels_onehot, labels, inf_rng):
    sstar = state.apply_fn({'params': params}, image, method='make_predictions')
    metrics = compute_metrics(image=image, labels_onehot=labels_onehot, labels=labels, state=state)
    return metrics

def eval_model(state, params, test_ds, batch_size, num_classes, eval_rng):
    t0 = time.time()
    test_ds_size = len(test_ds['image'])
    steps = test_ds_size // batch_size
    indices = jnp.arange(0, test_ds_size)
    indices = indices[:steps * batch_size]  # skip incomplete batch
    indices = indices.reshape((steps, batch_size))
    batch_metrics = []

    for idx in indices:
        batch = {k: v[idx, ...] for k, v in test_ds.items()}
        labels = batch['label']
        labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
        eval_rng, _ = jax.random.split(eval_rng)
        metrics = eval_step(state, params, batch['image'],labels_onehot, labels, eval_rng)
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    test_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}
    summary = jax.tree_util.tree_map(lambda x: x.item(), test_metrics_np)

    runtime = time.time() - t0

    # dummy states, used by get_L_and_gamma to infer correct array shape when generating random arrays
    sdummy = state.apply_fn({'params': params}, batch["image"], method='make_predictions')
    eval_rng, _ = jax.random.split(eval_rng)
    L10, gamma10 = state.apply_fn({'params':  state.params}, s=sdummy, rng_key=eval_rng, numiter=10, method='get_L_and_gamma')
    L20, gamma20 = state.apply_fn({'params':  state.params}, s=sdummy, rng_key=eval_rng, numiter=20, method='get_L_and_gamma')

    return summary['loss'], summary['accuracy'], summary['top5accuracy'], runtime, L10, L20, gamma10, gamma20

def ref_grad_and_angle(state, grads, image, labels_onehot, metrics):
    # Compute a reference backprop gradient (but don't use it).
    def loss_fn_ref(params, state):
        logits = state.apply_fn({'params': params}, image)
        loss = state.apply_fn({'params': params}, logits, labels_onehot, method='output_loss')
        return loss
    # compute logits, gradients and loss and accuracy.
    grad_fn_ref = jax.value_and_grad(loss_fn_ref, argnums=0, has_aux=False)
    _, ref_grads = grad_fn_ref(state.params, state)
    # Compute cosine sim between gradient and reference BP gradient.
    cosine_sim = cosine_sim_tree(grads, ref_grads)
    metrics["cosine_sim"] = cosine_sim
    return metrics

def no_ref_grad_and_angle(state, grads, image, labels_onehot, metrics):
    metrics["cosine_sim"] = jax.numpy.array([0.0]*len(state.params.keys()))
    return metrics


def cosine_sim_tree(grad_dict1, grad_dict2):
    cosine_sim_dict = {}
    for key in grad_dict1.keys():
        flattened_params_1, _ = jax.flatten_util.ravel_pytree(grad_dict1[key])
        flattened_params_1 = flattened_params_1.astype(jnp.float32)
        flattened_params_2, _ = jax.flatten_util.ravel_pytree(grad_dict2[key])
        flattened_params_2 = flattened_params_2.astype(jnp.float32)
        norm1 = jnp.linalg.norm(flattened_params_1)
        norm2 = jnp.linalg.norm(flattened_params_2)
        epsilon = jax.numpy.array(1e-8, dtype=jnp.float32)

        numerator = jnp.dot(flattened_params_1, flattened_params_2)
        denominator = jnp.max(jnp.array([norm1 * norm2, epsilon])) # for numerical stability
        cosine_sim_dict[key] = numerator/denominator
    
    # We exploit that the layers are named and the Dicts are ordered (conv00, conv01, ..., dense00, dense01, ...)
    # So the entries of the produced 1D array will have this same order
    cosine_sim_array, _ = jax.flatten_util.ravel_pytree(cosine_sim_dict)
    return cosine_sim_array

def heatmap_grads_batches(grad_cos_sim_batches, save_path, convert_to_angle, color_norm=None):

    if convert_to_angle == True:
        ax = sns.heatmap(np.arccos(np.clip(grad_cos_sim_batches, -1, 1))*180/np.pi, cmap="viridis", cbar_kws={'label': 'Angle (deg)'}, norm=color_norm,
                             yticklabels=np.array([i for i in range(1, 1+grad_cos_sim_batches.shape[0])]))
    else:
        ax = sns.heatmap(grad_cos_sim_batches, cmap="viridis", cbar_kws={'label': 'Cosine-sim'}, norm=color_norm,
                             yticklabels=np.array([i for i in range(1, 1+grad_cos_sim_batches.shape[0])]))
    figsize = np.array([10, 2])
    plt.gcf().set_size_inches(figsize[0], figsize[1])
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Layer")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return

def heatmap_grads_epochs(grad_cos_sim_epochs, save_path, convert_to_angle, color_norm=None):

    if convert_to_angle == True:
        ax = sns.heatmap(np.arccos(np.clip(grad_cos_sim_epochs, -1, 1))*180/np.pi, cmap="viridis", cbar_kws={'label': 'Angle (deg)'}, norm=color_norm,
                         yticklabels=np.array([i for i in range(1, 1+grad_cos_sim_epochs.shape[0])]))
    else:
        ax = sns.heatmap(grad_cos_sim_epochs, cmap="viridis", cbar_kws={'label': 'Cosine-sim'}, norm=color_norm,
                         yticklabels=np.array([i for i in range(1, 1+grad_cos_sim_epochs.shape[0])]))
    figsize = np.array([10, 2])
    plt.gcf().set_size_inches(figsize[0], figsize[1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return

def plot_L_or_gamma(L20, L10, ylabel, save_path):
    cm = plt.get_cmap('viridis')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    NUM_COLORS = L10.shape[0]
    ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    ax.plot(L20.T, lw=3, label=[f"layer {i+1}" for i in range(0,L10.shape[0]) ], alpha=0.8)
    ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    ax.plot(L10.T, ls="dashed", lw=2, alpha=1.0)
    ymax = 1.4*np.maximum(np.max(L20), np.max(L10))
    ax.set_ylim(0, ymax)
    figsize = np.array([5, 2.5])

    plt.gcf().set_size_inches(figsize[0], figsize[1])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend(fancybox=True, loc="upper left", fontsize='small', ncols=4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return