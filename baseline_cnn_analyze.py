import argparse
import datetime
import logging
import os

# from pyretina_systemidentification.models.regular_cnn import create_regular_cnn_model
# from pyretina_systemidentification.models.regular_cnn import create_klindt_cnn_model
from pyretina_systemidentification.models.regular_cnn import create_ecker_cnn_model
from data import Dataset


__basename__ = os.path.basename(__file__)
__name__, _ = os.path.splitext(__basename__)
__time__ = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                                                                   #usual command line argument acquisition
# Parse input arguments.
parser = argparse.ArgumentParser(description='baseline')
parser.add_argument(
    '-ei', '--experiment-identifier',
    nargs='?', type=str, const=None, default=__time__
)
# parser.add_argument(
#     '-cn', '--cell-number',
#     nargs='?', type=int, const=None, default=None
# )
parser.add_argument(
    '-rn', '--run-number',
    nargs='?', type=int, const=None, default=None
)
args = parser.parse_args()


# Set experiment identifier.
experiment_identifier = args.experiment_identifier
assert experiment_identifier is not None  # TODO correct!


# cell_nb = args.cell_number
# assert cell_nb is not None  # TODO correct!
cell_nb = None


run_nb = args.run_number


# Set up experiment directory.
experiment_directory = "{}".format(experiment_identifier)
assert os.path.isdir(experiment_directory), experiment_directory


# Set up logging.                                              #wrong logging to fix
log_filename = "{}_{}.log".format(__name__, __time__)
log_path = os.path.join(experiment_directory, log_filename)
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


# Prepare the data.                                       #loading all the cells from the pickle file
dataset = Dataset.load()
dataset.select_cells('all')
train_x, train_y = dataset.train()
val_x, val_y = dataset.val()
test_x, test_y = dataset.test()


# Create model.                                       #here are to be chosen the model parameters. Clear from the theory introduction
model_kwargs = {
    "core": {
        "nbs_kernels": (4,),
        "kernel_sizes": (31,),  # TODO correct?
        "strides": (1,),
        "paddings": ('valid',),  # TODO correct?
        "dilation_rates": (1,),
        "activations": ('relu',),  # TODO correct?
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


# Load TensorBoard scalars.
# run_name = "run_{:03d}".format(run_nb)
# # print(model_handler.get_tensorboard_scalars(run_name))
# tensorboard_tensors = model_handler.get_tensorboard_tensors(run_name)
# print(tensorboard_tensors)
# # print(tensorboard_tensors["ln_model/filter"]["value"].shape)
# # print(tensorboard_tensors["ln_model/filter"]["value"][-1])
# # print(tensorboard_tensors["ln_model/parametric_softplus"]["value"].shape)
# # print(tensorboard_tensors["ln_model/parametric_softplus"]["value"][-1])
# # print(tensorboard_tensors["batch_2"]["value"].shape)
# # print(tensorboard_tensors["batch_2"]["value"][-1])
# print(tensorboard_tensors["val_poisson"]["value"].shape)
# nb_runs = 2
#nb_runs = 128
# nb_runs = 64
nb_runs = 384

test_data = dataset.test(averages=False)
model_handler.get_randomized_search_table(nb_runs, test_data=test_data, force=False)    #here the fitted parameters of the model are loaded

val_data = dataset.val()
model_handler.plot_randomized_search_summary(nb_runs, val_data=val_data, test_data=test_data, train_data=train_data)


if run_nb is None:
    exit()
                   #IF WHEN YOU CALL THE PROGRAM YOU DO NOT SPECIFY A RUN, THE PROGRAM ENDS HERE. NOT VERY USEFUL

# Load model.
run_name = "run_{:03d}".format(run_nb)
model_handler.load(run_name=run_name)      #here you load the parameters of the specified run


# Save all predictions.
train_data = dataset.train()
val_data = dataset.val()
test_x, test_y = dataset.test(averages=False)
test_data = (test_x, test_y[0, :, :])  # i.e. keep one repetition only
model_handler.save_responses(
    train_data,
    val_data,
    test_data,
    filename="responses.xlsx",
)



# Save predictions on testing set.
test_x, test_y = dataset.test(averages=False)
test_data = (test_x, test_y)
model_handler.save_predictions(                 #here it saves the predictions of the model on the test.
    test_data,                                         #for details look in pyret-syst/models/regular_cnn
    observations_filename="test_observations.npy",
    predictions_filename="test_predictions.npy",
)

# #save predictions on training set
# train_x, train_y = dataset.train()
# train_data = (train_x, train_y)
# model_handler.save_predictions_train(                 #here it saves the predictions of the model on the test.
#     train_data,                                         #for details look in pyret-syst/models/regular_cnn
#     observations_filename="train_observations.npy",
#     predictions_filename="train_predictions.npy", )


'''
# Plot predictions.
test_x, test_y = dataset.test(averages=False)
test_data = (test_x, test_y)
model_handler.plot_predictions(
    test_data,
    highlighted_cells=[3, 5, 11, 19]
)
# TODO change / complete?
'''

# Create model table.
model_handler.get_model_table(test_data, force=True)

# Save LSTAs.
kwargs = {
    "lazy": True,
}
# model_handler.save_lstas(train_data, filename="train_lstas.npy", **kwargs)
# model_handler.save_lstas(val_data, filename="val_lstas.npy", **kwargs)
model_handler.save_lstas(test_data, filename="test_lstas.npy", **kwargs)
model_handler.save_lstas(test_data, filename="lstas.npy", append_grey_image=True, **kwargs)  # backward compatibility

########---------------NOT ALWAYS NEED THEM BUT IF NEEDED THEY ARE HERE 
#model_handler.save_lstas(train_data, filename="train_lstas.npy", **kwargs)
#model_handler.save_lstas(val_data, filename="validation_lstas.npy", **kwargs)
#######------------------------------------------

# Plot convolutional kernels.
model_handler.plot_convolutional_kernels()     #the fitted convolutional kernels of the core are saved as images

# # Plot first layer nonlinearities.
# fln = model_handler.get_first_layer_nonlinearities()
# # NB: ReLu used (without parameters) ...

# Plot spatial masks.
model_handler.plot_spatial_masks()

# Plot feature weights.
model_handler.plot_feature_weights()   #highlighted_cell_nbs=[0,1,2,3]
'''
# Plot model summary.
model_handler.plot_model_summary(test_data=test_data)

# Plot local STAs.
# model_handler.plot_local_stas(test_data)  # deprecated
model_handler.plot_local_stas(
    test_data,
    nb_columns=8,
    sta_ellipse_parameters_path="miscellaneous/sta_ellipse_parameters.csv",
    pixel_size=8.0*3.0e-6
)


#raise NotImplementedError("TODO")

'''
# TODO clean the following lines?


# Evaluate the model (with Numpy arrays).
test_data = (test_x, test_y)
#losses = model_handler.evaluate(test_data)
#logger.debug("losses: {}".format(losses))


# Predict with the model (with Numpy arrays).
predictions = model_handler.predict(test_data)            #HERE IT IS THE COMMAND TO OBTAIN THE MODEL PERFORMANCES
# logger.debug("predictions: {}".format(predictions))
logger.debug("predictions.shape: {}".format(predictions.shape))


# Compute reference losses.
import numpy as np
# Validation reference loss values.
val_x, val_y = dataset.val()
nb_images, nb_cells = val_y.shape
true_y = val_y
pred_y = np.mean(val_y, axis=(0,))  # i.e. mean over images
pred_y = np.broadcast_to(pred_y, val_y.shape)
# poisson_loss = np.mean(pred_y - true_y * np.log(pred_y))
poisson_loss = np.mean(pred_y - true_y * np.log(np.maximum(pred_y, 1e-16)))
print(poisson_loss)
# Test reference loss values.
test_x, test_y = dataset.test(averages=False)
nb_repetitions, nb_images, nb_cells = test_y.shape
true_y = np.reshape(test_y, (nb_repetitions * nb_images, nb_cells))
# ...
pred_y = np.mean(test_y, axis=(0, 1))  # i.e. mean over repetitions, images
pred_y = np.broadcast_to(pred_y, test_y.shape)
pred_y = np.reshape(pred_y, (nb_repetitions * nb_images, nb_cells))
# poisson_loss = np.mean(pred_y - true_y * np.log(pred_y))
poisson_loss = np.mean(pred_y - true_y * np.log(np.maximum(pred_y, 1e-16)))
print(poisson_loss)
# ...
pred_y = np.mean(test_y, axis=(0,))  # i.e. mean over repetitions
pred_y = np.broadcast_to(pred_y, test_y.shape)
pred_y = np.reshape(pred_y, (nb_repetitions * nb_images, nb_cells))
poisson_loss = np.mean(pred_y - true_y * np.log(np.maximum(pred_y, 1e-16)))
print(poisson_loss)


#nl=model_handler.get_readout_nonlinearities()

#print("This is the value of the bias averaged over cells:{0}".format(nl.mean()))

# # Plot predictions.
# test_x, test_y = dataset.test(averages=False)
# test_data = (test_x, test_y)
# model_handler.plot_predictions(
#     test_data,
# )
# # TODO change / complete?


# # Plot model evaluation.
# test_x, test_y = dataset.test(averages=False)
# test_data = (test_x, test_y)
# # cells_with_clear_polarity_inversions = [3, 5, 11, 19]
# cells_with_clear_polarity_inversions = None
# model_handler.plot_evaluation(
#     test_data,
#     highlighted_cells=cells_with_clear_polarity_inversions,
# )
# # TODO change / complete?


#TODO uncomment the following lines!
# if not is_gpu_available:

#     # Plot model elements.
#     if experiment_directory is not None:
#         model_handler.plot_convolutional_kernels(tag='initial')
#         model_handler.plot_spatial_masks(tag='initial')
#         model_handler.plot_feature_weights(tag='initial')
#     model_handler.plot_convolutional_kernels(tag='final')
#     model_handler.plot_spatial_masks(tag='final')
#     model_handler.plot_feature_weights(tag='final')

#     if experiment_directory is None:
#         plt.show()

'''
test_x, _ = dataset.test()
logger.debug("test_x.shape: {}".format(test_x.shape))
# nb_images, nb_rows, nb_columns, nb_channels = test_x.shape
# image_nb = 0
image_nb = 1
image = test_x[image_nb:image_nb+1, :, :, :]


# Plot activation map.
model_handler.plot_activation_map(image)


# Plot gradient.
model_handler.plot_gradient(image)
'''