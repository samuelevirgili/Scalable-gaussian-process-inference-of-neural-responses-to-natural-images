import argparse
import datetime
import logging
import os
import tensorflow as tf

# from pyretina_systemidentification.models.regular_cnn import create_regular_cnn_model
# from pyretina_systemidentification.models.regular_cnn import create_klindt_cnn_model
from pyretina_systemidentification.models.regular_cnn import create_ecker_cnn_model
from data import Dataset


__basename__ = os.path.basename(__file__)
__name__, _ = os.path.splitext(__basename__)
__time__ = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# Parse input arguments.
parser = argparse.ArgumentParser(description='baseline')
parser.add_argument(
    '-ei', '--experiment-identifier',
    nargs='?', type=str, const=None, default=__time__
)
args = parser.parse_args()


# Set log level for TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # i.e. keep all message
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # i.e. filter out info messages


# Set experiment identifier.
experiment_identifier = args.experiment_identifier


# Set up experiment directory.
if experiment_identifier is not None:
    experiment_directory = "{}".format(experiment_identifier)
    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)
else:
    experiment_directory = None


# Set up logging.
if experiment_directory is not None:
    log_filename = "{}_{}.log".format(__name__, __time__)
    log_path = os.path.join(experiment_directory, log_filename)
else:
    log_path = None
log_level = logging.DEBUG
# #
logging.basicConfig(
    filename=log_path,
    level=log_level,
)
# # Prevent Matplotlib from using the same log level.
matplotlib_logger = logging.getLogger(name='matplotlib')
matplotlib_logger.setLevel(logging.INFO)
# # Get logger.
logger = logging.getLogger(name=__name__)


# Check available CPUs and GPUs.
cpus = tf.config.list_physical_devices(device_type='CPU')
gpus = tf.config.list_physical_devices(device_type='GPU')
#assert len(gpus) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(gpus[0], True)  #THIS IS NEEDED FOR MY GPU BECAUSE IT HAS A SMALL MEMORY BUT MIGHT NOT BE TRUE FOR OTHER GPUS SO COMMENT IT IN CASE
if cpus:
    print("There is a CPU available:")
    print(cpus)
else:
    print("There is no CPU available.")
if gpus:
    print("There is a GPU available:")
    print(gpus)
else:
    print("There is no GPU available.")
if gpus:
    is_gpu_available = True
else:
    is_gpu_available = False


# Prepare the data.
dataset = Dataset.load()
dataset.select_cells('all')
train_x, train_y = dataset.train()
val_x, val_y = dataset.val()
test_x, test_y = dataset.test()

print('CHECK!!!! Shape training set: {} /n'.format(train_x.shape))


# Create model.
model_kwargs = {
    "core": {
        "nbs_kernels": (4,),
        "kernel_sizes": (31,),  # TODO correct?
        "strides": (1,),
        "paddings": ('valid',),  # TODO correct?
        "dilation_rates": (1,),
        # "activations": ('relu',),  # TODO correct?
        "activations": ('softplus',),  # TODO correct?
        "smooth_factors": (0.001,),  # TODO correct?
        "sparse_factors": (None,),
        "name": 'core',
    },
    "readout": {
        "nb_cells": 41,
        "spatial_sparsity_factor": 0.0001,  # TODO correct?
        "feature_sparsity_factor": 0.1,  # TODO correct?
        "name": 'readout',
    },
}
train_data = (train_x, train_y)
# model = create_regular_cnn_model(train_data=train_data, name="model")  # TODO remove?
# model = create_klindt_cnn_model(train_data=train_data, name="model")  # TODO remove?
model = create_ecker_cnn_model(model_kwargs=model_kwargs, train_data=train_data, name="model")
logger.debug("model.weights: {}".format(model.weights))


# Get model handler.
model_handler = model.create_handler(
    directory=experiment_directory,
    train_data=(train_x, train_y),
    val_data=(val_x, val_y),
    test_data=(test_x, test_y),
)


# # Random search.
# hyperparameters = {
#     'core/smooth_factors': ([1.0e-3, 1.0e+4],),
#     # 'core/sparse_factors': ([1.0e-6, 1.0e+1],),
#     'readout/spatial_sparsity_factor': [5.0e-3, 3.0e-2],  # c.f. [Ecker et al., 2018]  # incorrect
#     'readout/feature_sparsity_factor': [5.0e-3, 3.0e-2],  # c.f. [Ecker et al., 2018]  # incorrect
# }  # i.e. 20200612155116
# hyperparameters = {
#     'core/smooth_factors': ([1.0e-6, 1.0e+1],),
#     # 'core/sparse_factors': ([1.0e-6, 1.0e+1],),
#     'readout/spatial_sparsity_factor': [1.0e-5, 1.0e-1],
#     'readout/feature_sparsity_factor': [1.0e-5, 1.0e-1],
# }  # i.e. 20200615192009
# hyperparameters = {
#     'core/smooth_factors': ([1.0e-6, 1.0e+1],),
#     # 'core/sparse_factors': ([1.0e-6, 1.0e+1],),
#     'readout/spatial_sparsity_factor': [1.0e-6, 1.0e-2],
#     'readout/feature_sparsity_factor': [1.0e-6, 1.0e+1],
# }  # TODO correct?
hyperparameters = {
    'core/smooth_factors':  ([1.0e-10, 1.0e-1],), #([1.0e-7, 1.0e+1],),
    # 'core/sparse_factors': ([1.0e-6, 1.0e+1],),
    'readout/spatial_sparsity_factor':  [1.0e-10, 1.0e-1], #[1.0e-7, 1.0e-1],
    'readout/feature_sparsity_factor':  [1.0e-10, 1.0e-1],  #[1.0e-6, 1.0e0],
}  # TODO correct?
train_data = (train_x, train_y)
val_data = (val_x, val_y)
# nb_runs = 2
# nb_runs = 2 if not is_gpu_available else 32  # TODO correct?
# nb_runs = 2 if not is_gpu_available else 64
nb_runs = 128 if not is_gpu_available else 384  # TODO correct?
history = model_handler.randomized_search(
    hyperparameters,
    train_data,
    val_data,
    is_gpu_available=is_gpu_available,
    nb_runs=nb_runs,
)
