import argparse
import jax.numpy as jnp
from flax.linen import Conv, Dense, relu, tanh, sigmoid
from src import cnn_dualprop_Lagr_ff, cnn_dualprop_RAOVR_ff, cnn_dualprop_RAOVR_dampened_ff, cnn_abstract
from src import get_mnist, get_svhn, get_fashionmnist, get_cifar10, get_cifar100, get_imagenet_32x32
import optax
import jax.lax as lax
import jax
from math import prod

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--num-epochs', default=20, type=int, help='')

parser.add_argument('--batch-size', default=256, type=int, help='')

parser.add_argument('--learning-rate', default=0.015, type=float, help='')

parser.add_argument('--learning-rate-final', default=2e-6, type=float, help='')

parser.add_argument('--warmup-learning-rate', default=0.015, type=float, help='')

parser.add_argument('--decay-epochs', default=None, type=int, help='')

parser.add_argument('--warmup-epochs', default=0, type=int, help='')

parser.add_argument('--momentum', default=0.9, type=float, const=None, action='store', nargs='?', help='')

parser.add_argument('--weight-decay', default=5e-4, type=float, help='')

parser.add_argument('--seeds', nargs='+', default=[220], type=int, help='')

parser.add_argument('--percent-train',  default=95, type=int, help='How much of the training data to use for training. Only relevant for Imagenet datasets.')

parser.add_argument('--percent-val',  default=5, type=int, help='How much of the training data to use for validation. Only relevant for Imagenet datasets.')

parser.add_argument('--inference-sequence', default='fwbwK', choices=['fwbwK', 'fwK', 'bwK', 'oddeven', 'evenodd'], help='specify the update scheme to employ during training.')

parser.add_argument('--inference-passes-nudged', default=15, type=int, help='Number of update passes to employ.')

parser.add_argument('--experiment-name', default='test', help='A string denoting the name of the experiment. A directory with this name will be created. If the directory already exists you will get an error (to avoid accidentally overwriting existing expperiments).')

parser.add_argument('--model', default='VGG16', choices=['VGG16', 'VGGlike', 'CNN', 'miniCNN', 'MLP'], help='')

parser.add_argument('--learning-algorithm', default='dualprop-lagr-ff', choices=['backprop', 'dualprop-lagr-ff', 'dualprop-raovr-ff', 'dualprop-raovr-dampened-ff'])

dtypes = {'bfloat16': jnp.bfloat16, 'float16':jnp.float16, 'float32':jnp.float32}
parser.add_argument('--dtype', default='float32', choices=dtypes.keys())
parser.add_argument('--param-dtype', default='float32', choices=['bfloat16', 'float16', 'float32'])

parser.add_argument('--beta', default=1, type=float)

parser.add_argument('--alpha', default=0.5, type=float)

parser.add_argument('--activation', default='relu', choices=['relu', 'hs', 'sigmoid', 'tanh'])

losses = {"sce":optax.softmax_cross_entropy, "mse":optax.squared_error}
parser.add_argument('--loss', default='sce', choices=losses.keys())

datasets = dict(fashionmnist=get_fashionmnist, mnist=get_mnist, svhn=get_svhn, cifar10=get_cifar10, cifar100=get_cifar100, imagenet_32x32=get_imagenet_32x32)
parser.add_argument('--dataset', '-d', choices=datasets.keys(), default='cifar10', help='')

config = parser.parse_args()

# Note for comparisson to previous work we pad MNIST and FMNIST to size (32,32,1)
imagedims = {"fashionmnist": (28,28,1), "mnist": (32,32,1), "svhn": (32,32,3), "cifar10": (32,32,3), "cifar100": (32,32,3), "imagenet_32x32": (32,32,3)}
config.image_dims = imagedims[config.dataset]
num_classes = {"fashionmnist": 10, "mnist": 10, "svhn": 10, "cifar10": 10, "cifar100": 100, "imagenet_32x32": 1000}
config.num_classes = num_classes[config.dataset]
config.dtype = dtypes[config.dtype]
config.param_dtype = dtypes[config.param_dtype]

@jax.jit
def hs(x):
    return jnp.minimum(1, jnp.maximum(x, 0))

def loss_func(logits, one_hot):
    logits = logits.astype(jnp.float32)
    one_hot = one_hot.astype(jnp.float32)
    return jnp.sum(losses[config.loss](logits, one_hot))

# Choose architecture
if config.model == "VGG16":
    kernels = [(3,3), (3,3), (3,3), (3,3),(3,3), (3,3), (3,3), (3,3),(3,3), (3,3), (3,3), (3,3), (3,3)]
    strides = [(1,1), (1,1), (1,1), (1,1),(1,1), (1,1), (1,1), (1,1),(1,1), (1,1), (1,1), (1,1), (1,1)]
    mp = [False, True, False, True, False, False, True, False, False, True, False, False, True]
    features = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    dense_features = [4096, 4096, config.num_classes]
elif config.model == "VGGlike":
    kernels = [(3,3), (3,3), (3,3), (3,3)]
    strides = [(1,1), (1,1), (1,1), (1,1)]
    mp = [True, True, True, True]
    features = [128, 256, 512, 512]
    dense_features = [config.num_classes]
elif config.model == "CNN":
    kernels = [(3,3), (3,3), (3,3), (3,3), (3,3)]
    strides = [(1,1), (1,1), (1,1), (1,1), (1,1)]
    mp = [True, True, True, True, True]
    features = [64, 128, 256, 512, 512]
    dense_features = [4096, config.num_classes]
elif config.model == "miniCNN":
    kernels = [(3,3), (3,3)]
    strides = [(1,1), (1,1)]
    mp = [True, True]
    features = [64, 64]
    dense_features = [config.num_classes]
elif config.model == "MLP":
    kernels = []
    strides = []
    mp = []
    features = []
    dense_features = [1024, 1024, config.num_classes]

# Load model
modeltype = {"backprop":cnn_abstract, "dualprop-lagr-ff": cnn_dualprop_Lagr_ff, "dualprop-raovr-ff": cnn_dualprop_RAOVR_ff, "dualprop-raovr-dampened-ff": cnn_dualprop_RAOVR_dampened_ff}
activation={"relu": relu, "hs": hs, "sigmoid": sigmoid, "tanh": tanh}
config.model = modeltype[config.learning_algorithm](loss_func, Conv, Dense, activation[config.activation], config.num_classes, config.beta, config.alpha, config.dtype, config.param_dtype,
                                    kernels=kernels, strides=strides, features=features, mp = mp,
                                    dense_features=dense_features, inference_sequence=config.inference_sequence, inference_passes_nudged=config.inference_passes_nudged
                                    )

# Load datasets
config.train_ds, config.val_ds, config.test_ds = datasets[config.dataset](config.dtype, config.percent_train, config.percent_val)