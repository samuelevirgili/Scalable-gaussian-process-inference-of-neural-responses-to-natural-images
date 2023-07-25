import copy
import logging
import matplotlib.patches as pcs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import tensorflow as tf

from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyretina_systemidentification.cores.stacked_convolution_2d import StackedConv2DCore
from pyretina_systemidentification.readouts.spatial_feature_joint_l1 import SpatialXFeatureJointL1Readout
from pyretina_systemidentification.callbacks.reduce_learning_rate_on_plateau import CustomReduceLearningRateOnPlateauCallback  # noqa
from pyretina_systemidentification.callbacks.model_checkpoint import CustomModelCheckpointCallback
from pyretina_systemidentification.utils import corrcoef, deepupdate
from pyretina_systemidentification.utils import plot_scale_bar


logger = logging.getLogger(__name__)


# Model.

class RegularCNNModel(tf.keras.Model):
    """Regular CNN model."""

    model_default_kwargs = {
        "core": {
            "nbs_kernels": (4,),
            "kernel_sizes": (21,),
            "strides": (1,),
            "paddings": ('valid',),
            "dilation_rates": (1,),
            "activations": ('relu',),
            "smooth_factors": (0.001,),  # (0.01,),
            "sparse_factors": (None,),
            "name": 'core',
        },
        "readout": {
            "nb_cells": 63,  # TODO correct!
            "x": None,
            "y": None,
            "spatial_masks_initializer": 'truncated normal',  # TODO correct!
            "feature_weights_initializer": 'truncated normal',  # TODO correct!
            "non_negative_feature_weights": True,  # TODO try `True` instead?
            "spatial_sparsity_factor": 0.0001,  # TODO use another value (default `0.01`)?
            "feature_sparsity_factor": 0.1,  # TODO use another value (default `0.01`)?
            "name": 'readout',
        }
    }

    def __init__(
            self, model_kwargs=None, learning_rate=0.002, train_data=None, name="model", **kwargs
    ):
        """Initialization of the model."""

        super().__init__(name=name, **kwargs)

        # Model keyword arguments.
        self.model_kwargs = copy.deepcopy(self.model_default_kwargs)
        if self.model_kwargs is not None:
            self.model_kwargs = deepupdate(self.model_kwargs, model_kwargs)
        if train_data is not None:
            train_x, train_y = train_data
            self.model_kwargs = deepupdate(
                self.model_kwargs,
                {
                    "readout": {
                        "x": train_x,
                        "y": train_y,
                    }
                }
            )
        # ...
        self.learning_rate = learning_rate
        # Initialize core.
        core_kwargs = self.model_kwargs["core"]
        self.core = StackedConv2DCore(**core_kwargs)
        # Initialize readout.
        readout_kwargs = self.model_kwargs["readout"]
        self.readout = SpatialXFeatureJointL1Readout(**readout_kwargs)

    def compile(self, **kwargs):
        """Configure the learning process of the model."""

        if self._is_compiled:
            logger.warning("Model has already been compiled.")
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                # beta_1=0.9,
                # beta_2=0.999,
                # epsilon=1e-07,
                # amsgrad=False,
                name='Adam',
                # **kwargs,
            )
            loss = tf.keras.losses.Poisson(
                # reduction=losses_utils.ReductionV2.AUTO,
                name='poisson'
            )
            metrics = [
                tf.keras.metrics.Poisson(
                    name='poisson',
                    # dtype=None
                )
            ]
            super().compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                # loss_weights=None,
                # sample_weight_mode=None,
                # weighted_metrics=None,
                # target_tensors=None,
                # distribute=None,
                # **kwargs
            )

        return

    def call(self, inputs, training=False, **kwargs):
        """Forward computation of the model."""

        internals = self.core(inputs, training=training)
        outputs = self.readout(internals)

        return outputs

    def create_handler(self, *args, **kwargs):

        return _RegularCNNModelHandler(self, *args, **kwargs)


create_regular_cnn_model = RegularCNNModel  # i.e. alias


# Klindt CNN model.

class KlindtCNNModel(RegularCNNModel):

    model_default_kwargs = {
        "core": {
            "nbs_kernels": (4,),
            "kernel_sizes": (21,),
            "strides": (1,),
            "paddings": ('valid',),
            "dilation_rates": (1,),
            "activations": ('relu',),
            "smooth_factors": (0.001,),  # (0.01,),
            "sparse_factors": (None,),
            "name": 'core',
        },
        "readout": {
            "nb_cells": 63,  # TODO correct!
            "x": None,
            "y": None,
            "spatial_masks_initializer": '[Klindt et al., 2017]',
            "feature_weights_initializer": '[Klindt et al., 2017]',
            "non_negative_feature_weights": True,  # TODO try `True` instead?
            "spatial_sparsity_factor": 0.0001,  # TODO use another value (default `0.01`)?
            "feature_sparsity_factor": 0.1,  # TODO use another value (default `0.01`)?
            "name": 'readout',
        },
    }

    def __init__(self, *args, learning_rate=0.001, **kwargs):

        super().__init__(*args, learning_rate=learning_rate, **kwargs)


create_klindt_cnn_model = KlindtCNNModel  # i.e. alias


# Ecker CNN model.

class EckerCNNModel(RegularCNNModel):

    model_default_kwargs = {
        "core": {
            "nbs_kernels": (4,),
            "kernel_sizes": (31,),
            "strides": (1,),
            "paddings": ('valid',),
            "dilation_rates": (1,),
            "activations": ('relu',),
            "smooth_factors": (0.001,),  # (0.01,),
            "sparse_factors": (None,),
            "name": 'core',
        },
        "readout": {
            "nb_cells": 114,  # TODO correct!
            "x": None,
            "y": None,
            "spatial_masks_initializer": '[Ecker et al., 2019]',
            "feature_weights_initializer": '[Ecker et al., 2019]',
            "non_negative_feature_weights": True,  # TODO try `True` instead?
            "spatial_sparsity_factor": 0.0001,  # TODO use another value (default `0.01`)?
            "feature_sparsity_factor": 0.1,  # TODO use another value (default `0.01`)?
            "name": 'readout',
        },
    }

    def __init__(self, *args, learning_rate=0.01, **kwargs):

        super().__init__(*args, learning_rate=learning_rate, **kwargs)


create_ecker_cnn_model = EckerCNNModel  # i.e. alias


# Model handler.

class _RegularCNNModelHandler:

    def __init__(self, model, directory=None, train_data=None, val_data=None, test_data=None):

        model.compile()

        self._model = model
        self._directory = directory
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data

        self._checkpoints = dict()
        self._tag = None
        self._run_name = None

    # @property
    # def _initial_weights_path(self):
    #
    #     if self._directory is not None:
    #         path = os.path.join(self._directory, "initial_model_weights")
    #     else:
    #         path = None
    #
    #     return path

    # @property
    # def _final_weights_path(self):
    #
    #     if self._directory is not None:
    #         path = os.path.join(self._directory, "final_model_weights")
    #     else:
    #         path = None
    #
    #     return path

    def _get_checkpoint_weights_path(self, tag='final', run_name=None):

        if self._directory is not None:
            # ...
            if run_name is not None:
                directory = os.path.join(self._directory, run_name)
            else:
                directory = self._directory
            # ...
            if tag is None:
                path = os.path.join(directory, "checkpoint_final")
            elif isinstance(tag, str):
                path = os.path.join(directory, "checkpoint_{}".format(tag))
            elif isinstance(tag, int):
                path = os.path.join(directory, "checkpoint_{:05d}".format(tag))
            else:
                raise TypeError("unexpected tag type: {}".format(type(tag)))
        else:
            path = None

        return path

    def train(self, train_data, val_data, epochs=None, is_gpu_available=False, run_name=None):

        train_x, train_y = train_data
        val_x, val_y = val_data

        # Evaluate model (to make the loading effective, to be able to save the initial weights).
        _ = self._model.evaluate(val_x, val_y, batch_size=32, verbose=0)

        # Save initial weights (if necessary).
        path = self._get_checkpoint_weights_path(tag='initial', run_name=run_name)
        if path is not None:
            self._model.save_weights(
                path,
                # overwrite=True,
                save_format='tf',  # or 'h5'?
            )

        # Infer/fit/train the model (with Numpy arrays).
        # # Prepare callbacks.
        monitor = 'val_poisson'  # i.e. not 'val_loss'
        callbacks = []
        # # # Enable checkpoints.
        if self._directory is not None:
            if run_name is None:  # TODO simplify!
                checkpoint_path = os.path.join(self._directory, "checkpoint_{epoch:05d}")
            else:
                checkpoint_path = os.path.join(self._directory, run_name, "checkpoint_{epoch:05d}")
            callback = CustomModelCheckpointCallback(
                checkpoint_path,
                monitor=monitor,
                verbose=1,  # {0 (quiet, default), 1 (update messages), 2 (update and debug messages)}
                save_best_only=True,  # False (default)
                save_weights_only=True,  # False (default)
                save_max=2,  # 10 max.
                # mode='auto',  # {'auto', 'min', 'max'}
                # save_freq='epoch',
                # **kwargs,
            )
            callbacks.append(callback)
        # # # Enable TensorBoard and image summaries.
        if self._directory is not None:
            if run_name is None:  # TODO simplify!
                tensorboard_path = os.path.join(self._directory, "logs")
            else:
                tensorboard_path = os.path.join(self._directory, "logs", run_name)
            # Enable TensorBoard.
            callback = tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_path,
                histogram_freq=1,  # in epochs  {0 (no computation, default), <integer> (computation)}
                write_graph=True,
                # write_images=False,
                # update_freq='epoch',  # {'batch', 'epoch' (default), <integer>} note that writing too frequently to TensorBoard can slow down the training  # noqa
                # profile_batch=2,  # {0 (disable), 2 (default), <integer>}
                # embeddings_freq=0,  # in epochs  {0 (no visualized), <integer> (visualized)}
                # embeddings_metadata=None,
                # **kwargs,
            )
            callbacks.append(callback)
            # # # Enable image summaries.
            import io
            def plot_to_image(figure):  # noqa
                """Converts the matplotlib plot specified by 'figure' to a PNG image and
                returns it. The supplied figure is closed and inaccessible after this call."""
                # Save the plot to a PNG in memory.
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                # Closing the figure prevents it from being displayed directly inside
                # the notebook.
                plt.close(figure)
                buf.seek(0)
                # Convert PNG buffer to TF image
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                # Add the batch dimension
                image = tf.expand_dims(image, 0)
                return image
            def prepare_kernels_plot(kernels):  # noqa
                _, _, nb_features, nb_kernels = kernels.shape
                fig, axes = plt.subplots(nrows=nb_features, ncols=nb_kernels, squeeze=False)
                imshow_kwargs = {
                    'cmap': 'RdBu_r',
                    'vmin': -tf.math.reduce_max(tf.math.abs(kernels)),
                    'vmax': tf.math.reduce_max(tf.math.abs(kernels)),
                }
                for kernel_nb in range(0, nb_kernels):
                    kernel = kernels[:, :, :, kernel_nb]
                    for feature_nb in range(0, nb_features):
                        kernel_slice = kernel[:, :, feature_nb]  # TODO use `kernel_map` instead of `kernel_slice`?
                        ax = axes[feature_nb, kernel_nb]
                        ax.imshow(kernel_slice, **imshow_kwargs)
                        # TODO check the 2 previous lines.
                fig.tight_layout()
                return fig
            def prepare_spatial_mask_plot(spatial_mask):  # noqa
                fig, ax = plt.subplots()
                imshow_kwargs = {
                    'cmap': 'Greys_r',
                    'vmin': 0.0,
                    'vmax': tf.math.reduce_max(spatial_mask),
                }
                ax.imshow(spatial_mask, **imshow_kwargs)
                fig.tight_layout()
                return fig
            image_summaries_path = os.path.join(tensorboard_path, "train")
            image_summaries_writer = tf.summary.create_file_writer(image_summaries_path)
            def log_image_summaries(epoch):  # noqa
                if epoch is None or epoch < 10 - 1:  # i.e. the maximum number of images visible in tensorboard in 10.
                    with image_summaries_writer.as_default():
                        step = epoch if epoch is not None else 10000
                        # Log core convolutional kernels.
                        for layer in self._model.core.convolution_layers:
                            kernels = layer.kernel
                            logger.debug("kernels.shape: {}".format(kernels.shape))
                            tf.summary.image(
                                kernels.name,
                                plot_to_image(prepare_kernels_plot(kernels)),
                                step=step,
                            )
                        # TODO uncomment the following lines (or gather spatial masks in a single figure)?
                        # # Log readout spatial masks.
                        # spatial_masks = self._model.readout.masks
                        # logger.debug("spatial_masks.shape: {}".format(spatial_masks.shape))
                        # nb_cells, _, _ = spatial_masks.shape
                        # for cell_nb in range(0, nb_cells):
                        #     tf.summary.image(
                        #         "{}:{}".format(spatial_masks.name, cell_nb),
                        #         plot_to_image(prepare_spatial_mask_plot(spatial_masks[cell_nb, :, :])),
                        #         step=step,
                        #     )
                        # TODO log readout feature masks.
                return
            callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: log_image_summaries(epoch),
                on_train_end=lambda logs: log_image_summaries(None),
            )
            callbacks.append(callback)
        # # # Enable learning rate decays.
        learning_rate_decay_factor = 0.5  # TODO use `0.1` instead (default)?
        # TODO set a `minimum_learning_rate` to limit the number of learning rate decays?
        callback = CustomReduceLearningRateOnPlateauCallback(
            monitor=monitor,
            factor=learning_rate_decay_factor,
            patience=10,
            verbose=1,  # {0 (quiet, default), 1 (update messages)}
            # mode='auto',  # {auto (default), min, max}
            # min_delta=0.0001,  # threshold for measuring the new optimum, to only focus on significant changes
            # cooldown=0,  # number of epochs to wait before resuming normal operation after the learning rate has been reduced  # noqa
            # min_lr=0,  # lower bound on the learning rate (default: 0)
            restore_best_weights=True,  # TODO understand why TensorFlow does not implement this by default???
            # **kwargs,
        )
        callbacks.append(callback)
        # # # Enable early stopping.
        callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            # min_delta=0,
            patience=20,  # use `0` instead (default)?
            verbose=1,  # {0 (quiet?, default), 1 (update messages?)}
            # mode='auto',
            # baseline=None,
            restore_best_weights=True,
        )
        callbacks.append(callback)
        # # Run the inference of the model.
        batch_size = 32 #64  # 128  # 256  # 32
        if epochs is None:
            epochs = 1000 if is_gpu_available else 150
        verbose = 2 if is_gpu_available else 2  # verbosity mode (0 = silent, 1 = progress bar (interactive environment), 2 = one line per epoch (production environment))  # noqa
        history = self._model.fit(
            train_x,
            train_y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            # validation_split=0.0,
            validation_data=(val_x, val_y),
            # shuffle=True,
            # class_weight=None,
            # sample_weight=None,
            # initial_epoch=0,  # useful for resuming a previous training session
            # steps_per_epoch=None,
            # validation_steps=None,
            # validation_freq=1,
            # max_queue_size=10,
            # workers=1,
            # use_multiprocessing=False,
            # **kwargs
        )

        # Save final weights (if necessary).
        path = self._get_checkpoint_weights_path(tag='final', run_name=run_name)
        if path is not None:
            self._model.save_weights(
                path,
                # overwrite=True,
                save_format='tf',  # or 'h5'?
            )

        # Update attributes.
        self._tag = 'final'
        self._run_name = run_name

        return history

    def _convert_domain(self, domain):

        from tensorboard.plugins.hparams import api as hp  # TODO move to top of file?

        if domain is None:
            values = ['None']
            dtype = None  # TODO correct or remove!
            converted_domain = hp.Discrete(values, dtype=dtype)
        elif isinstance(domain, (int, float, str)):
            values = [domain]
            dtype = None  # TODO correct of remove!
            converted_domain = hp.Discrete(values, dtype=dtype)
        elif isinstance(domain, tuple):
            converted_domain = tuple([
                self._convert_domain(sub_domain)
                for sub_domain in domain
            ])  # TODO avoid `ValueError: not a domain: (RealInterval(0.002, 0.04), RealInterval(0.002, 0.04))`!
        elif isinstance(domain, set):
            values = list(domain)
            dtype = None  # TODO correct or remove!
            converted_domain = hp.Discrete(values, dtype=dtype)
        elif isinstance(domain, list):
            assert len(domain) == 2, domain
            min_value = domain[0]
            max_value = domain[1]
            if isinstance(min_value, float) and isinstance(max_value, float):
                assert min_value <= max_value, domain
                converted_domain = hp.RealInterval(min_value=min_value, max_value=max_value)
            elif isinstance(min_value, int) and isinstance(max_value, int):
                assert min_value <= max_value, domain
                converted_domain = hp.IntInterval(min_value=min_value, max_value=max_value)
            else:
                raise TypeError(
                    "unexpected min_value ({}) and max_value types({})".format(type(min_value), type(max_value))
                )
        else:
            # TODO correct!
            raise TypeError("unexpected domain type ({})".format(type(domain)))

        return converted_domain

    def _convert_hyperparameters(self, hyperparameters):
        """Hyperparameters conversion (from dict to TensorBoard API)"""

        from tensorboard.plugins.hparams import api as hp  # TODO move to top of file?

        assert isinstance(hyperparameters, dict), hyperparameters

        converted_hyperparameters = dict()
        for name, domain in hyperparameters.items():
            converted_domain = self._convert_domain(domain)
            if isinstance(converted_domain, tuple):
                for k, converted_sub_domain in enumerate(converted_domain):
                    assert not isinstance(converted_sub_domain, tuple)  # TODO implement?
                    sub_name = name + "_{}".format(k)
                    converted_hyperparameters[sub_name] = hp.HParam(sub_name, domain=converted_sub_domain)
            else:
                converted_hyperparameters[name] = hp.HParam(name, domain=converted_domain)

        return converted_hyperparameters

    def _sample_domain(self, domain):

        if domain is None:
            sampled_value = domain
        elif isinstance(domain, (int, float, str)):
            sampled_value = domain
        elif isinstance(domain, tuple):
            sampled_value = tuple([
                self._sample_domain(sub_domain)
                for sub_domain in domain
            ])
        elif isinstance(domain, set):
            values = list(domain)
            sampled_value = random.choice(values)
        elif isinstance(domain, list):
            assert len(domain) == 2, domain
            min_value = domain[0]
            max_value = domain[1]
            if isinstance(min_value, float) and isinstance(max_value, float):
                assert min_value <= max_value, domain
                # sampled_value = random.uniform(min_value, max_value)  # i.e. uniform
                sampled_value = np.exp(random.uniform(np.log(min_value), np.log(max_value)))  # i.e. log-uniform
            elif isinstance(min_value, int) and isinstance(max_value, int):
                assert min_value <= max_value, domain
                sampled_value = random.randint(min_value, max_value)
            else:
                raise TypeError(
                    "unexpected min_value ({}) and max_value types({})".format(type(min_value), type(max_value))
                )
        else:
            # TODO correct!
            raise TypeError("unexpected domain type ({})".format(type(domain)))

        return sampled_value

    def _sample_hyperparameters(self, hyperparameters, seed=None):  # TODO move outside class?

        assert isinstance(hyperparameters, dict), hyperparameters

        if seed is None:
            # random.seed(a=None)  # i.e. use current system time to initialize the random number generator.
            pass  # TODO correct?
        else:
            random.seed(a=seed)

        sampled_hyperparameters = dict()
        for name, domain in hyperparameters.items():
            sampled_value = self._sample_domain(domain)
            sampled_hyperparameters[name] = sampled_value

        return sampled_hyperparameters

    def randomized_search(self, hyperparameters, train_data, val_data, is_gpu_available=False, nb_runs=2):
        """Hyperparameter optimization/tuning."""

        from tensorboard.plugins.hparams import api as hp  # TODO move to top of file?

        if self._directory is not None:
            tensorboard_path = os.path.join(self._directory, "logs")  # TODO move to class.
            hparams_summary_path = tensorboard_path  # TODO rename?
        else:
            hparams_summary_path = None

        if not os.path.isdir(hparams_summary_path):  # i.e. search has not already been ran

            if hparams_summary_path is not None:
                # Log the experiment configuration to TensorBoard.
                converted_hyperparameters = self._convert_hyperparameters(hyperparameters)
                with tf.summary.create_file_writer(hparams_summary_path).as_default():
                    hp.hparams_config(
                        hparams=list(converted_hyperparameters.values()),
                        metrics=[
                            hp.Metric('val_loss', display_name='val_loss'),
                            hp.Metric('val_poisson', display_name='val_poisson'),
                        ],
                        # time_created_secs=None,  # i.e. current time (default)
                    )

            for run_nb in range(0, nb_runs):

                run_name = "run_{:03d}".format(run_nb)

                # Sample a random combination of hyperparameters.
                sampled_hyperparameters = self._sample_hyperparameters(hyperparameters, seed=run_nb)
                # Sanity prints.
                print("Run {:03d}/{:03d}:".format(run_nb, nb_runs))
                for name, value in sampled_hyperparameters.items():
                    print("    {}: {}".format(name, value))

                # Create model.
                model_kwargs = copy.deepcopy(self._model.model_kwargs)
                # # Update hyperparameters involved in this random search.
                for name, value in sampled_hyperparameters.items():
                    keys = name.split('/')
                    kwargs = model_kwargs
                    for key in keys[:-1]:
                        kwargs = kwargs[key]
                    kwargs[keys[-1]] = value
                # # Clear TF graph (i.e. use same namespace).
                tf.keras.backend.clear_session()
                # # Instantiate & compile model.
                model = self._model.__class__(
                    model_kwargs=model_kwargs,
                    train_data=self._train_data,
                    name="model",
                )
                model.compile()
                self._model = model

                # Train model.
                history = self.train(train_data, val_data, is_gpu_available=is_gpu_available, run_name=run_name)

                if hparams_summary_path is not None:
                    # run_summary_path = os.path.join(hparams_summary_path, run_name)
                    run_summary_path = os.path.join(hparams_summary_path, run_name, "train")
                    # Log the hyperparameters and metrics to TensorBoard.
                    with tf.summary.create_file_writer(run_summary_path).as_default():
                        # Log hyperparameter values for the current run/trial.
                        formatted_hyperparameters = dict()
                        for name, value in sampled_hyperparameters.items():
                            if value is None:
                                formatted_value = 'None'
                                formatted_hyperparameters[name] = formatted_value
                            elif isinstance(value, tuple):
                                for k, sub_value in enumerate(value):
                                    sub_name = name + "_{}".format(k)
                                    if sub_value is None:
                                        formatted_value = 'None'
                                        formatted_hyperparameters[sub_name] = formatted_value
                                    else:
                                        formatted_hyperparameters[sub_name] = sub_value
                            else:
                                formatted_hyperparameters[name] = value
                        _ = hp.hparams(
                            formatted_hyperparameters,
                            trial_id=run_name,
                            # start_time_secs=None,  # i.e. current time
                        )
                        # Log hyperparameters for programmatic use.
                        for name, value in formatted_hyperparameters.items():
                            name = "hyperparameters/{}".format(name)
                            value = value.item()  # i.e. numpy.float to float
                            tf.summary.scalar(name, value, step=0, description=None)
                        # Log metrics.
                        for step, val_loss in enumerate(history.history['val_loss']):
                            tf.summary.scalar('val_loss', val_loss, step=step)
                        for step, val_poisson in enumerate(history.history['val_poisson']):
                            tf.summary.scalar('val_poisson', val_poisson, step=step)
                        # TODO use callbacks instead of writing these directly?

        else:

            logger.debug("randomized search has already been ran")

        # TODO reload best model!

        return

    def load(self, tag=None, run_name=None):
        """Load model."""

        # Load trained weights (if possible).
        try:
            self._model.load_weights(
                self._get_checkpoint_weights_path(tag=tag, run_name=run_name),
                # by_name=False,
                # skip_mismatch=False,
            ).expect_partial()
            # c.f. https://stackoverflow.com/questions/58289342/tf2-0-translation-model-error-when-restoring-the-saved-model-unresolved-objec.  # noqa
            self._tag = tag
            self._run_name = run_name
        except (tf.errors.NotFoundError, ValueError):
            raise FileNotFoundError

        # Test modtl (if possible, to make the loading effective).
        if self._test_data is not None:
            test_x, test_y = self._test_data
            _ = self._model.evaluate(test_x, test_y, batch_size=32, verbose=0)
        else:
            raise NotImplementedError("TODO")

        return

    def evaluate(self, test_data, batch_size=32):  # TODO correct `batch_size`?
        """Evaluate model."""

        test_x, test_y = test_data
        losses = self._model.evaluate(
            test_x,
            test_y,
            batch_size=batch_size,
        )

        return losses

    def predict(self, test_data, batch_size=32):  # TODO correct `batch_size`?
        """Predict model output."""

        test_x, test_y = test_data
        predictions = self._model.predict(
            test_x,
            batch_size=batch_size,
        )
        if test_y.ndim == predictions.ndim + 1:
            # Add dimension for repetitions.
            reps = (test_y.shape[0],) + (test_y.ndim - 1) * (1,)
            predictions = np.tile(predictions, reps)
        assert predictions.shape == test_y.shape, (predictions.shape, test_y.shape)

        return predictions

    def predict_local_spike_triggered_averages(self, test_data):

        test_x, _ = test_data
        stimulus = tf.Variable(tf.cast(test_x, tf.float32))
        logger.debug("stimulus.shape: {}".format(stimulus.shape))
        response = self._model(stimulus)
        logger.debug("response.shape: {}".format(response.shape))
        batch_size, nb_cells = response.shape
        gradients = nb_cells * [None]
        for cell_nb in range(0, nb_cells):
            with tf.GradientTape() as tape:
                response = self._model(stimulus)
                gradients[cell_nb] = tape.gradient(response[:, cell_nb], stimulus)
                # TODO normalize gradient?
        # logger.debug("gradients: {}".format(gradients))
        logger.debug("type(gradients): {}".format(type(gradients)))
        gradients = np.array([
            g.numpy()
            for g in gradients
        ])
        logger.debug("type(gradients): {}".format(type(gradients)))
        logger.debug("gradients.shape: {}".format(gradients.shape))
        # nb_cells, nb_batches, nb_rows, nb_columns, nb_channels = gradients.shape
        lstas = gradients[:, :, :, :, 0]

        return lstas

    predict_lstas = predict_local_spike_triggered_averages  # i.e. alias

    def predict_lstas2(self, test_data, cell_idx):

        test_x, _ = test_data
        stimulus = tf.Variable(tf.cast(test_x, tf.float32))
        logger.debug("stimulus.shape: {}".format(stimulus.shape))
        response = self._model(stimulus)
        logger.debug("response.shape: {}".format(response.shape))
        batch_size, nb_cells = response.shape
        gradients = 1 * [None]
        
        with tf.GradientTape() as tape:
            response = self._model(stimulus)
            gradients[0] = tape.gradient(response[:, cell_idx], stimulus)
            # TODO normalize gradient?
        # logger.debug("gradients: {}".format(gradients))
        logger.debug("type(gradients): {}".format(type(gradients)))
        gradients = np.array([
            g.numpy()
            for g in gradients
        ])
        logger.debug("type(gradients): {}".format(type(gradients)))
        logger.debug("gradients.shape: {}".format(gradients.shape))
        # nb_cells, nb_batches, nb_rows, nb_columns, nb_channels = gradients.shape
        lstas = gradients[0, :, :, :, 0]

        return lstas


    def _get_checkpoint(self, tag='final', run_name=None):

        if tag in self._checkpoints:
            model = self._checkpoints[tag]
        else:
            # model = create_regular_cnn_model(train_data=self._train_data, name="{}_model".format(tag))  # TODO remove?
            model = self._model.__class__(train_data=self._train_data, name="{}_model".format(tag))  # TODO correct (pass keyword arguments)!
            model.compile()
            weights_path = self._get_checkpoint_weights_path(tag=tag, run_name=run_name)
            if weights_path is not None:
                try:
                    model.load_weights(
                        weights_path,
                        # by_name=False,
                        # skip_mismatch=False,
                    )
                except tf.errors.NotFoundError:
                    path = weights_path + ".index"
                    raise FileNotFoundError(path) from None
                if self._test_data is not None:
                    test_x, test_y = self._test_data
                    _ = model.evaluate(test_x, test_y, batch_size=32, verbose=0)
                else:
                    raise NotImplementedError()
            else:
                pass
            self._checkpoints[tag] = model

        return model

    def get_tensorboard_scalars(self, run_name):

        assert self._directory is not None, self._directory

        # Load event multiplexer.
        from tensorboard.backend.event_processing import event_multiplexer
        size_guidance = {
            "distributions": 500,
            "images": 4,
            "audio": 4,
            "scalars": 10000,
            "histograms": 1,
            # "tensors": 10,
            "tensors": 1000,
        }
        em = event_multiplexer.EventMultiplexer(
            # run_path_map=None,
            size_guidance=size_guidance,
            # purge_orphaned_data=True,
        )
        directory = os.path.join(self._directory, "logs")
        em.AddRunsFromDirectory(
            directory,
            # name=None
        )
        logger.debug("event_multiplexer.Runs(): {}".format(em.Runs()))

        # Get event accumulator.
        ea = em.GetAccumulator("{}/train".format(run_name))
        # ea = em.GetAccumulator("{}/validation".format(run_name))
        ea.Reload()

        # Get scalars.
        scalar_keys = ea.scalars.Keys()
        logger.debug("Scalar keys: {}".format(scalar_keys))
        scalars = dict()
        for scalar_key in scalar_keys:
            scalars[scalar_key] = dict()
            scalar_events = ea.Scalars(scalar_key)
            scalars[scalar_key]['wall_time'] = np.array([
                scalar_event.wall_time
                for scalar_event in scalar_events
            ])
            scalars[scalar_key]['step'] = np.array([
                scalar_event.step
                for scalar_event in scalar_events
            ])
            scalars[scalar_key]['value'] = np.array([
                scalar_event.value
                for scalar_event in scalar_events
            ])

        return scalars

    def get_tensorboard_tensors(self, run_name):

        assert self._directory is not None, self._directory

        # Load event multiplexer.
        from tensorboard.backend.event_processing import event_multiplexer
        size_guidance = {
            "distributions": 500,
            "images": 4,
            "audio": 4,
            "scalars": 10000,
            "histograms": 1,
            # "tensors": 10,
            "tensors": 1000,
        }
        em = event_multiplexer.EventMultiplexer(
            # run_path_map=None,
            size_guidance=size_guidance,
            # purge_orphaned_data=True,
        )
        directory = os.path.join(self._directory, "logs")
        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)
        em.AddRunsFromDirectory(
            directory,
            # name=None
        )
        logger.debug("Run keys: {}".format(em.Runs()))

        # Get event accumulator.
        ea = em.GetAccumulator("{}/train".format(run_name))
        # ea = em.GetAccumulator("{}/validation".format(run_name))
        ea.Reload()

        # Get tensors.
        tensor_keys = ea.tensors.Keys()
        logger.debug("Tensor keys: {}".format(tensor_keys))
        tensors = dict()
        for tensor_key in tensor_keys:
            tensors[tensor_key] = dict()
            tensor_events = ea.Tensors(tensor_key)
            tensors[tensor_key]['wall_time'] = np.array([
                tensor_event.wall_time
                for tensor_event in tensor_events
            ])
            tensors[tensor_key]['step'] = np.array([
                tensor_event.step
                for tensor_event in tensor_events
            ])
            tensors[tensor_key]['value'] = np.array([
                tf.make_ndarray(tensor_event.tensor_proto)
                for tensor_event in tensor_events
            ])

        return tensors

    def get_randomized_search_table(self, nb_runs, test_data=None, force=False):

        if self._directory is not None:
            path = os.path.join(self._directory, "randomized_search.csv")
        else:
            path = None

        if path is None or not os.path.isfile(path) or force:
            # Collect hyperparameter names.
            import re
            tensorboard_tensors = self.get_tensorboard_tensors("run_000")
            hyperparameter_names = []
            for name in tensorboard_tensors.keys():
                if re.search("^hyperparameters/", name):
                    hyperparameter_names.append(name)
                    # hyperparameter_names.append(name.replace("hyperparameters/", ""))
            logger.debug(hyperparameter_names)
            # Collect data.
            hyperparameters = {
                hyperparameter_name: []
                for hyperparameter_name in hyperparameter_names
            }
            val_poissons = []
            val_losses = []
            for run_nb in range(0, nb_runs):
                run_name ="run_{:03d}".format(run_nb)
                tensorboard_tensors = self.get_tensorboard_tensors(run_name)
                # print(tensorboard_tensors.keys())  # TODO remove!
                # print(run_nb)
                for hyperparameter_name in hyperparameter_names:
                    key = hyperparameter_name
                    # key = "hyperparameters/{}".format(hyperparameter_name)
                    hyperparameters[hyperparameter_name].append(tensorboard_tensors[key]["value"][0])
                val_poissons.append(tensorboard_tensors["val_poisson"]["value"][-1])
                val_losses.append(tensorboard_tensors["val_loss"]["value"][-1])
            data = {
                key: np.array(value)
                for key, value in hyperparameters.items()
            }
            data.update({
                'val_poisson': np.array(val_poissons),
                'val_losses': np.array(val_losses),
            })
            # Collect additional data (if possible).
            if test_data is not None:
                pass  # TODO complete (i.e. 'test_poisson' and 'test_accuracy')!
            # Create data frame.
            df = pd.DataFrame(data)
            # Save to file.
            df.to_csv(
                path,
                index=True,  # i.e. write row names
            )
            print("Randomized search table saved to:")
            print("  {}".format(path))
        else:
            # Load from file.
            df = pd.read_csv(
                path,
                index_col=0,
            )

        return df

    def get_model_table(self, test_data, force=False):

        if self._directory is not None and self._run_name is not None:
            path = os.path.join(self._directory, self._run_name, "performances.csv")
        else:
            path = None

        if path is None or not os.path.isfile(path) or force:

            _, test_y = test_data
            # test_y = np.mean(test_y, axis=0)  # i.e. mean over repetitions
            # nb_conditions, nb_cells = test_y.shape
            # pred_y = self.predict(test_data)
            nb_repetitions, nb_conditions, nb_cells = test_y.shape
            pred_y = self.predict(test_data)

            even_test_y = np.mean(test_y[0::2, :, :], axis=0)  # i.e. mean over even repetitions
            odd_test_y = np.mean(test_y[1::2, :, :], axis=0)  # i.e. mean over odd repetitions
            # ...
            reliabilities = np.empty(nb_cells, dtype=np.float)
            accuracies = np.empty(nb_cells, dtype=np.float)
            total_variances = np.empty(nb_cells, dtype=np.float)
            observation_noise_variances = np.empty(nb_cells, dtype=np.float)
            explainable_variances = np.empty(nb_cells, dtype=np.float)
            explainable_variance_fractions = np.empty(nb_cells, dtype=np.float)
            # TODO rename (explainable_to_total_variance_ratios)?
            mean_squared_errors = np.empty(nb_cells, dtype=np.float)
            explained_variances = np.empty(nb_cells, dtype=np.float)
            explained_variance_fractions = np.empty(nb_cells, dtype=np.float)
            # TODO rename (explained_to_explainable_variance_ratios)?
            explained_variance_fractions_bis = np.empty(nb_cells, dtype=np.float)
            # TODO rename?
            # ...
            for cell_nb in range(0, nb_cells):
                reliabilities[cell_nb] = corrcoef(even_test_y[:, cell_nb], odd_test_y[:, cell_nb])
                accuracies[cell_nb] = corrcoef(np.mean(pred_y[:, :, cell_nb], axis=0), odd_test_y[:, cell_nb])
                total_variances[cell_nb] = np.var(test_y[:, :, cell_nb])
                observation_noise_variances[cell_nb] = np.mean(np.var(test_y[:, :, cell_nb], axis=0))
                # i.e. variance over repetitions and mean over conditions
                explainable_variances[cell_nb] = total_variances[cell_nb] - observation_noise_variances[cell_nb]
                explainable_variance_fractions[cell_nb] = explainable_variances[cell_nb] / total_variances[cell_nb]
                # TODO rename?
                mean_squared_errors[cell_nb] = np.mean(np.square(test_y[:, :, cell_nb] - pred_y[:, :, cell_nb]))
                explained_variances[cell_nb] = total_variances[cell_nb] - mean_squared_errors[cell_nb]
                explained_variance_fractions[cell_nb] = explained_variances[cell_nb] / explainable_variances[cell_nb]
                # TODO rename?
                explained_variance_fractions_bis[cell_nb] = accuracies[cell_nb] / reliabilities[cell_nb]
                # TODO rename?
            # Create data frame.
            data = {
                "reliability": reliabilities,
                "accuracy": accuracies,
                "total_variance": total_variances,
                "observation_noise_variance": observation_noise_variances,
                "explainable_variances": explainable_variances,
                "explainable_variance_fraction": explainable_variance_fractions,
                # TODO rename?
                "mean_squared_errors": mean_squared_errors,
                "explained_variance": explained_variances,
                "explained_variance_fraction": explained_variance_fractions,
                # TODO rename?
                "explained_variance_fraction_bis": explained_variance_fractions_bis,
                # TODO rename?
                # TODO complete?
            }
            df = pd.DataFrame(data)
            # Save to file.
            df.to_csv(
                path,
                index=True,  # i.e. write row names
                index_label="cell_nb",
            )
            print("Model table saved to:")
            print("  {}".format(path))
        else:
            # Load from file.
            df = pd.read_csv(
                path,
                index_col="cell_nb",
            )

        return df

    # Getters.

    def get_convolutional_kernels(self):
        """Get convolutional kernels."""

        model = self._model
        # model = self._get_checkpoint(tag=tag)
        weights = {
            weight.name: weight
            for weight in model.weights
        }
        logger.debug(weights.keys())
        kernels = weights["model/core/conv_0/kernel:0"]
        kernels = kernels.numpy()
        # TODO reshape?

        return kernels

    def get_core_nonlinearities(self):

        model = self._model
        # model = self._get_checkpoint(tag=tag)
        weights = {
            weight.name: weight
            for weight in model.weights
        }
        logger.debug(weights.keys())

        # ReLu (without any parameter)
        # Softplus (without any parameter)

        raise NotImplementedError()

    def get_spatial_masks(self):
        """Get spatial masks."""

        model = self._model
        # model = self._get_checkpoint(tag=tag)
        weights = {
            weight.name: weight
            for weight in model.weights
        }
        logger.debug(weights.keys())
        masks = weights["model/readout/masks:0"]
        masks = masks.numpy()
        # TODO reshape?

        return masks

    def get_feature_weights(self):
        """Get feature weights."""

        model = self._model
        # model = self._get_checkpoint(tag=tag)
        weights = {
            weight.name: weight
            for weight in model.weights
        }
        logger.debug(weights.keys())
        features = weights["model/readout/feature_weights:0"]
        features = features.numpy()
        # TODO reshape?

        return features

    def get_readout_nonlinearities(self):

        model = self._model
        # model = self._get_checkpoint(tag=tag)
        weights = {
            weight.name: weight
            for weight in model.weights
        }
        logger.debug(weights.keys())
        readout_nl_params = weights["model/readout/biases:0"]
        readout_nl_params = readout_nl_params.numpy()
        # Softplus (without any parameter but a bias)

        return readout_nl_params

    # ...

    def save_responses(self, train_data, val_data, test_data, filename="responses.xlsx"):

        train_x, train_y = train_data
        val_x, val_y = val_data
        test_x, test_y = test_data

        train_y_pred = self.predict(train_data)
        val_y_pred = self.predict(val_data)
        test_y_pred = self.predict(test_data)

        observed_spike_count = np.concatenate((train_y, val_y, test_y))
        predicted_spike_count = np.concatenate((train_y_pred, val_y_pred, test_y_pred))

        assert observed_spike_count.shape == predicted_spike_count.shape, \
            (observed_spike_count.shape, predicted_spike_count.shape)

        nb_conditions, nb_cells = observed_spike_count.shape
        data_frame = {
            cell_id: pd.DataFrame.from_dict({
                "observed_spike_count": observed_spike_count[:, cell_id],
                "predicted_spike_count": predicted_spike_count[:, cell_id],
            })
            for cell_id in range(0, nb_cells)
        }

        path = os.path.join(self._directory, self._run_name, filename)
        with pd.ExcelWriter(path) as writer:
            for cell_id in range(0, nb_cells):
                data_frame[cell_id].to_excel(writer, sheet_name="cell_{:02d}".format(cell_id))
        print("Responses saved to {}".format(path))

        return

    # ...

    def save_predictions(self, data, observations_filename="observations.npy", predictions_filename="predictions.npy"):

        x, y = data
        nb_repetitions, nb_conditions, nb_cells = y.shape
        # print(y.shape)
        y_ = np.mean(y, axis=0)  # i.e. mean over repetitions
        # nb_conditions, nb_cells = y_.shape
        y_pred_ = self.predict((x, y_))
        # nb_conditions, nb_cells = y_pred.shape
        y_pred = np.tile(y_pred_, (nb_repetitions, 1, 1))  # i.e. reconstruction dimension of repetitions
        # nb_repetitions, nb_conditions, nb_cells = y_pred.shape
        # print(y_pred.shape)

        path = os.path.join(self._directory, self._run_name, observations_filename)
        np.save(path, y)
        print("Observations saved to {}".format(path))

        path = os.path.join(self._directory, self._run_name, predictions_filename)
        np.save(path, y_pred)
        print("Predictions saved to {}".format(path))

        return

    def save_predictions_train(self, data, observations_filename="train_observations.npy", predictions_filename="train_predictions.npy"):

        x, y = data
        nb_conditions, nb_cells = y.shape
        y_pred = self.predict((x, y))
        # print(y_pred.shape)

        path = os.path.join(self._directory, self._run_name, observations_filename)
        np.save(path, y)
        print("Observations saved to {}".format(path))

        path = os.path.join(self._directory, self._run_name, predictions_filename)
        np.save(path, y_pred)
        print("Predictions saved to {}".format(path))

        return

    def save_local_spike_triggered_averages(self, data, filename="lstas.npy", append_grey_image=False, lazy=False):

        assert self._directory is not None
        assert self._run_name is not None
        path = os.path.join(self._directory, self._run_name, filename)

        if not lazy or not os.path.isfile(path):

            x, _ = data
            if append_grey_image:
                grey_value = np.mean(x)
                print(grey_value)  # TODO comment!
                grey_shape = (1,) + x.shape[1:]
                grey_image = grey_value * np.ones(grey_shape)
                grey_image = grey_image.astype(x.dtype)
                x = np.concatenate((x, grey_image))
                data = x, None
            print("x.shape: {}".format(x.shape))  # TODO remove!
            nb_images, height, width, nb_channels = x.shape
            batch_size = 64
            if nb_images <= batch_size:
                lstas = self.predict_lstas(data)
            else:
                from tqdm import tqdm
                print("batch size: {} images".format(batch_size))
                nb_batches = (nb_images - 1) // batch_size + 1
                lstas = [
                    self.predict_lstas((x[batch_size*batch_nb:batch_size*(batch_nb+1)], None))
                    for batch_nb in tqdm(range(0, nb_batches), unit="batch")
                ]
                # nb_cells, batch_size, height, width = lstas[0].shape
                lstas = np.concatenate(lstas, axis=1)
            print(lstas.shape)  # TODO comment!
            # nb_cells, nb_images, height, width = lstas.shape

            # Save to disk.
            np.save(path, lstas)

        else:

            import warnings
            warnings.warn("Found {}, won't recompute the LSTAs (lazy mode activated)...".format(path))

            # Load from disk.
            lstas = np.load(path)

        return lstas

    save_local_stas = save_local_spike_triggered_averages  # i.e alias
    save_lstas = save_local_spike_triggered_averages  # i.e. alias

    # Plotters.

    def _get_plot_path(self, with_run_name=True):

        if self._directory is not None:
            path = os.path.join(self._directory, "plots")
            if with_run_name and self._run_name is not None:
                path = os.path.join(path, self._run_name)
            if not os.path.isdir(path):
                os.makedirs(path)
        else:
            path = None

        return path

    def plot_convolutional_kernels(self, nb_rows=1, pixel_size=8.0*3.0, scale=100.0, unit='µm'):
        """Plot convolutional kernels."""

        # Get kernels.
        kernels = self.get_convolutional_kernels()
        nb_vertical_pixels, nb_horizontal_pixels, _, nb_features = kernels.shape
        are_all_zeros = np.max(np.abs(kernels)) <= sys.float_info.epsilon

        # Plot kernels.
        # figsize = (float(nb_features) * 1.0 * 1.6, 1.0 * 1.6)
        figsize = (float(nb_features), 2.0)
        nb_columns = (nb_features - 1) // nb_rows + 1
        fig, axes = plt.subplots(nrows=nb_rows, ncols=nb_columns, squeeze=False, figsize=figsize)
        imshow_kwargs = {
            'cmap': 'RdBu_r',
            'vmin': -np.max(np.abs(kernels)),
            'vmax': +np.max(np.abs(kernels)),
            'extent': (0.0, float(nb_vertical_pixels) * pixel_size, 0.0, float(nb_horizontal_pixels) * pixel_size),
        }
        mat = None
        for feature_nb in range(0, nb_features):
            # Plot kernel.
            # ax = axes[0, feature_nb]
            ax = axes.flatten()[feature_nb]
            kernel = kernels[:, :, 0, feature_nb]
            mat = ax.matshow(kernel, **imshow_kwargs)
            ax.xaxis.set_ticks_position('bottom')  # i.e. move labels to bottom
            # # Set axis major locator.
            # # ax.xaxis.set_major_locator(plt.MaxNLocator(2))  # i.e. reduce number of ticks
            # # ax.yaxis.set_major_locator(plt.MaxNLocator(2))  # i.e. reduce number of ticks
            # height, width = kernel.shape
            # ax.xaxis.set_major_locator(plt.FixedLocator([0, width - 1]))
            # ax.yaxis.set_major_locator(plt.FixedLocator([0, height - 1]))
            # Set axis ticks.
            ax.set_xticks([])
            ax.set_yticks([])
            # Set title.
            ax.set_title("kernel {}".format(feature_nb))
            plot_scale_bar(ax, scale=scale, unit='µm', loc='lower left', with_label=(feature_nb == 0))
            # TODO add scale bar?
        # Tight layout.
        fig.tight_layout()
        # Add colorbar (if possible).
        if mat is not None and not are_all_zeros:
            ax = axes[nb_rows - 1, nb_columns - 1]
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax_ins = inset_axes(ax, width="5%", height="80%", loc='right')
            cb = fig.colorbar(mat, cax=ax_ins, orientation='vertical')
            # cb.set_label("w")
            ax_ins.yaxis.set_major_locator(plt.MaxNLocator(2))
            cb.set_ticks([])

        # Save plot (if necessary).
        plot_path = self._get_plot_path(with_run_name=True)
        if plot_path is not None:
            suffix = "_{}".format(self._tag) if self._tag is not None else ""
            output_filename = "convolutional_kernels{}.pdf".format(suffix)
            output_path = os.path.join(plot_path, output_filename)
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
        #save the kernels themselves as an array
        np.save(os.path.join(plot_path, "convolutional_kernels.npy"), kernels[:,:,0,:])

        return

    def plot_spatial_masks(self, pixel_size=8.0*3.0):
        """Plot spatial masks."""

        # Get masks.
        masks = self.get_spatial_masks()
        nb_cells, height, width = masks.shape
        are_all_zeros = np.max(masks) <= sys.float_info.epsilon

        # Plot masks.
        # nb_rows = 5  # TODO correct (auto.)!
        # nb_columns = 6  # TODO correct (auto.)!
        nb_rows = 9  # TODO correct (auto.)!
        nb_columns = 5  # TODO correct (auto.)!
        fig, axes = plt.subplots(
            nrows=nb_rows, ncols=nb_columns, squeeze=False,
            # gridspec_kw={
            #     'wspace': 0.05,
            #     'hspace': 0.05,
            # },
            # figsize=(3.0 * 1.6, 3.0 * 1.6)
            gridspec_kw={
                'wspace': 0.1,
                'hspace': 0.1,
            },
            # figsize=(float(nb_columns) * 0.8 * 1.6, float(nb_rows) * 0.8 * 1.6)
            figsize = (float(nb_columns), float(nb_rows))
        )
        for row_nb in range(0, nb_rows):
            for column_nb in range(0, nb_columns):
                ax = axes[row_nb, column_nb]
                ax.set_visible(False)
        imshow_kwargs = {
            'cmap': 'Greys_r',
            'vmin': 0.0,
            'vmax': np.max(masks) if not are_all_zeros else 0.1,
            'extent': (0.0, float(width) * pixel_size, 0.0, float(height) * pixel_size),
        }
        mat = None
        for cell_nb in range(0, nb_cells):
            ax = axes.ravel()[cell_nb]
            ax.set_visible(True)
            mask = masks[cell_nb, :, :]
            mat = ax.matshow(mask, **imshow_kwargs)
            ax.xaxis.set_ticks_position('bottom')  # i.e. move labels to bottom
            plot_scale_bar(ax, scale=300.0, unit='µm', loc='lower left', with_label=(cell_nb == 0), color='white')
        # Add annotations (i.e. cell identities).
        for cell_nb in range(0, nb_cells):
            ax = axes.ravel()[cell_nb]
            text_kwargs = {
                # 'alpha': 0.5,
                # 'backgroundcolor': 'white',
                # 'color': 'grey',
                'color': 'white',
                # 'bbox': {
                #     'boxstyle': 'round',
                #     'alpha': 0.3,
                #     # 'edgecolor': 'black',
                #     'facecolor': 'white',
                #     # 'linewidth': 0.3,
                #     'linewidth': 0.0,  # i.e. without edge
                # },
                'fontsize': 'small',  # or 'medium'
                'horizontalalignment': 'left',
                'verticalalignment': 'top',
            }
            ax.annotate(
                "c{:02d}".format(cell_nb), (0.0 + 0.05, 1.0 - 0.05), xycoords='axes fraction', **text_kwargs
            )

        # Handle axis labels and ticks.
        for row_nb in range(0, nb_rows):
            for column_nb in range(0, nb_columns):
                ax = axes[row_nb, column_nb]
                if row_nb == nb_rows - 1 and column_nb == 0:
                    # ax.xaxis.set_major_locator(plt.MaxNLocator(2))
                    # ax.yaxis.set_major_locator(plt.MaxNLocator(2))
                    # ax.xaxis.set_major_locator(plt.FixedLocator([0, width - 1]))
                    # ax.yaxis.set_major_locator(plt.FixedLocator([0, height - 1]))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # ax.set_xlabel("x")
                    # ax.set_ylabel("y")
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
        # Tight layout.
        fig.tight_layout()
        # Add colorbar (if possible).
        if mat is not None and not are_all_zeros:
            ax = axes[nb_rows - 1, nb_columns - 1]
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(ax, width="10%", height="80%", loc='right')
            cb = fig.colorbar(mat, cax=axins, orientation='vertical')
            # cb.set_label("w")
            axins.yaxis.set_major_locator(plt.MaxNLocator(2))

        # Save plot (if necessary).
        plot_path = self._get_plot_path(with_run_name=True)
        if plot_path is not None:
            suffix = "_{}".format(self._tag) if self._tag is not None else ""
            output_filename = "spatial_masks{}.pdf".format(suffix)
            output_path = os.path.join(plot_path, output_filename)
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)

            np.save(os.path.join(plot_path, "spatial_masks.npy"), masks)

        return

    def plot_feature_weights(self, highlighted_cell_nbs=None):
        """Plot feature weights."""

        # Get features.
        weights = self.get_feature_weights()
        nb_cells, nb_features = weights.shape
        are_all_zeros = np.max(weights) <= sys.float_info.epsilon

        highlighted_mask = np.zeros(nb_cells, dtype=np.bool)
        highlighted_mask[highlighted_cell_nbs] = True

        # Plot features.
        # fig, ax = plt.subplots(figsize=(1.5 * 1.6, 3.0 * 1.6))
        # fig, axes = plt.subplots(figsize=(1.0, 2.0))
        subplots_kwargs = {
            'ncols': 2,
            'squeeze': False,
            'gridspec_kw': {'wspace': 0.2},
            'figsize': (3.0, 1.5),
        }
        fig, axes = plt.subplots(**subplots_kwargs)
        # # 1st plot.
        ax = axes[0, 0]
        imshow_kwargs = {
            'cmap': 'Greys_r',
            'vmin': 0.0,
            'vmax': np.max(weights) if not are_all_zeros else 0.1,
        }
        mat = ax.matshow(weights, **imshow_kwargs)
        ax.xaxis.set_ticks_position('bottom')  # i.e. move labels to bottom
        # ax.xaxis.set_major_locator(plt.MaxNLocator(2))  # i.e. reduce number of ticks
        # ax.yaxis.set_major_locator(plt.MaxNLocator(2))  # i.e. reduce number of ticks
        ax.xaxis.set_major_locator(plt.FixedLocator([0, nb_features - 1]))  # i.e. reduce number of ticks
        ax.yaxis.set_major_locator(plt.FixedLocator([0, nb_cells - 1]))  # i.e. reduce number of ticks
        ax.set_xticks([]) if nb_features <= 4 else None
        ax.set_xlabel("feature")
        ax.set_ylabel("cell")
        # Add colorbar (if necessary).
        if not are_all_zeros:
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", 0.1, pad=0.1)
            # fig.colorbar(mat, cax=cax)
            # cax.yaxis.set_major_locator(plt.MaxNLocator(2))  # i.e. reduce number of ticks
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(
                ax, width=0.05, height="40%", loc='lower right',
                bbox_to_anchor=(2, 0, 1, 1), bbox_transform=ax.transAxes
            )
            cb = fig.colorbar(mat, cax=axins, orientation='vertical')
            # cb.set_label("w")
            axins.yaxis.set_major_locator(plt.MaxNLocator(2))
            cb.set_ticks([])
        # # 2nd plot.
        ax = axes[0, 1]
        x = weights[:, 0]
        y = weights[:, 1]
        scatter_kwargs = {
            's': 3 ** 2,
            'c': 'black',
        }
        ax.scatter(x[~highlighted_mask], y[~highlighted_mask], ec=None, **scatter_kwargs)
        ax.scatter(x[highlighted_mask], y[highlighted_mask], ec='tab:red', **scatter_kwargs)
        # Set axis limits.
        ax.set_xlim(left=-0.01)
        ax.set_ylim(bottom=-0.01)
        # Set axis tick locators.
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        # Set axis labels.
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
        # Hide the right and top spines
        for loc in ['right', 'top']:
            ax.spines[loc].set_visible(False)
        # Tight layout.
        fig.tight_layout()

        # Save plot (if necessary).
        plot_path = self._get_plot_path(with_run_name=True)
        if plot_path is not None:
            suffix = "_{}".format(self._tag) if self._tag is not None else ""
            output_filename = "feature_weights{}.pdf".format(suffix)
            output_path = os.path.join(plot_path, output_filename)
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)

        #save the feature weights themselves
        output_path_w = os.path.join(plot_path, "feature_weights.npy")
        np.save(output_path_w, weights)


        return

    def plot_predictions(self, *args, **kwargs):

        self.plot_evaluation(*args, **kwargs)

        return

    def plot_evaluation(self, test_data, highlighted_cells=None):
        """Plot model evaluation."""

        test_x, test_y = test_data
        test_y = np.mean(test_y, axis=0)  # i.e. mean over repetitions
        nb_conditions, nb_cells = test_y.shape
        # pred_y = self.predict(test_data)
        pred_y = self.predict((test_x, test_y))

        # Plot predictions vs responses (cell-wise).
        nb_rows = 5  # TODO correct (auto.)!
        nb_columns = 6  # TODO correct (auto.)!
        fig, axes = plt.subplots(
            nrows=nb_rows, ncols=nb_columns, squeeze=False,
            gridspec_kw={
                'wspace': 0.1,
                'hspace': 0.1,
            },
            figsize=(float(nb_columns) * 0.8 * 1.6, float(nb_rows) * 0.8 * 1.6)
        )
        for row_nb in range(0, nb_rows):
            for column_nb in range(0, nb_columns):
                ax = axes[row_nb, column_nb]
                ax.set_visible(False)
        scatter_kwargs = {
            's': 1.5 ** 2,
        }
        # v_min = min(np.min(pred_y), np.min(pred_y))
        v_min = 0.0
        margin_factor = 0.1
        v_max = max(np.max(pred_y), np.max(test_y))
        x_min = v_min - margin_factor * (v_max - v_min)
        x_max = v_max + margin_factor * (v_max - v_min)
        y_min = v_min - margin_factor * (v_max - v_min)
        y_max = v_max + margin_factor * (v_max - v_min)
        for cell_nb in range(0, nb_cells):
            ax = axes.ravel()[cell_nb]
            ax.set_visible(True)
            ax.set_aspect('equal')
            x = test_y[:, cell_nb]
            y = pred_y[:, cell_nb]
            if highlighted_cells is None:
                c = 'black'
            else:
                c = 'tab:blue' if cell_nb in highlighted_cells else 'black'
            ax.scatter(x, y, c=c, **scatter_kwargs)
            # Add identity line.
            ax.plot([v_min, v_max], [v_min, v_max], color='grey', linewidth=0.3)
            # Add annotation (i.e. cell identity).
            text_kwargs = {
                # 'alpha': 0.5,
                # 'backgroundcolor': 'white',
                'color': 'grey',
                'bbox': {
                    'boxstyle': 'round',
                    'alpha': 0.3,
                    # 'edgecolor': 'black',
                    'facecolor': 'white',
                    # 'linewidth': 0.3,
                    'linewidth': 0.0,  # i.e. without edge
                },
                'fontsize': 'x-small',  # 'x-small', 'small' or 'medium'
                'horizontalalignment': 'left',
                'verticalalignment': 'top',
            }
            ax.annotate(
                "c{:02d}".format(cell_nb), (0.0 + 0.05, 1.0 - 0.05), xycoords='axes fraction', **text_kwargs
            )
            # Add annotation (i.e. corr. coef.).
            corr_coef = corrcoef(x, y)  # TODO correct!
            if corr_coef < 0.5:
                color = 'tab:red'
            elif corr_coef < 0.85:
                color = 'tab:grey'
            else:
                color = 'tab:green'
            text_kwargs = {
                # 'alpha': 0.5,
                # 'backgroundcolor': 'white',
                'color': color,
                'bbox': {
                    'boxstyle': 'round',
                    'alpha': 0.3,
                    # 'edgecolor': 'black',
                    'facecolor': 'white',
                    # 'linewidth': 0.3,
                    'linewidth': 0.0,  # i.e. without edge
                },
                'fontsize': 'x-small',  # 'x-small', 'small' or 'medium'
                'horizontalalignment': 'left',
                'verticalalignment': 'top',
            }
            ax.annotate(
                "r:{:.3f}".format(corr_coef), (0.0 + 0.05, 1.0 - 0.20), xycoords='axes fraction', **text_kwargs
            )
            # TODO add annotation (i.e. fraction of explained variance).

            # Handle axis limits.
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        # Handle axis labels and ticks.
        for row_nb in range(0, nb_rows):
            for column_nb in range(0, nb_columns):
                ax = axes[row_nb, column_nb]
                # Set axes major locators.
                ax.xaxis.set_major_locator(plt.MaxNLocator(2))
                ax.yaxis.set_major_locator(plt.MaxNLocator(2))
                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if row_nb == nb_rows - 1 and column_nb == 0:
                    ax.set_xlabel("response\n(data)")
                    ax.set_ylabel("prediction\n(model)")
                else:
                    # ax.set_xticks([])
                    # ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
        fig.tight_layout()

        # Save plot (if necessary).
        plot_path = self._get_plot_path()
        if plot_path is not None:
            suffix = "_{}".format(self._tag) if self._tag is not None else ""
            output_filename = "evaluation_predictions_cellwise{}.pdf".format(suffix)
            output_path = os.path.join(plot_path, output_filename)
            fig.savefig(output_path)
            plt.close(fig)

        _, test_y = test_data
        # nb_repetitions, nb_conditions, nb_cells = test_y.shape

        # Plot accuracies vs reliabilities (population-wise).
        even_test_y = np.mean(test_y[0::2, :, :], axis=0)  # i.e. mean over even repetitions
        odd_test_y = np.mean(test_y[1::2, :, :], axis=0)  # i.e. mean over odd repetitions
        reliabilities = []
        accuracies = []
        for cell_nb in range(0, nb_cells):
            reliability = corrcoef(even_test_y[:, cell_nb], odd_test_y[:, cell_nb])
            accuracy = corrcoef(pred_y[:, cell_nb], odd_test_y[:, cell_nb])
            reliabilities.append(reliability)
            accuracies.append(accuracy)
        reliabilities = np.array(reliabilities)
        accuracies = np.array(accuracies)
        plot_path = self._get_plot_path()
        if plot_path is not None:
            suffix = "_{}".format(self._tag) if self._tag is not None else ""
            # Save reliabilities.
            output_filename = "evaluation_predictions_reliabilities{}.npy".format(suffix)
            output_path = os.path.join(plot_path, output_filename)
            np.save(output_path, reliabilities)
            # Save accuracies.
            output_filename = "evaluation_predictions_accuracies{}.npy".format(suffix)
            output_path = os.path.join(plot_path, output_filename)
            np.save(output_path, accuracies)
        # ...
        fig, ax = plt.subplots(
            figsize=(3.0 * 1.6, 3.0 * 1.6)
        )
        ax.set_aspect('equal')
        x = reliabilities
        y = accuracies
        scatter_kwargs = {
            's': 2 ** 2,
        }
        v_min = 0.0
        v_max = max(np.max(reliabilities), np.max(accuracies))
        margin_factor = 0.05
        x_min = v_min - margin_factor * (v_max - v_min)
        x_max = v_max + margin_factor * (v_max - v_min)
        y_min = v_min - margin_factor * (v_max - v_min)
        y_max = v_max + margin_factor * (v_max - v_min)
        if highlighted_cells is None:
            ax.scatter(x, y, c='black', **scatter_kwargs)
        else:
            ax.scatter(x, y, c='black', **scatter_kwargs)
            ax.scatter(x[highlighted_cells], y[highlighted_cells], c='red', **scatter_kwargs)
            # TODO find the proper way to do that!
        # Add point/cell labels.
        text_kwargs = {
            'color': 'grey',
            'bbox': {
                'boxstyle': 'round',
                'alpha': 0.3,
                # 'edgecolor': 'black',
                'facecolor': 'white',
                # 'linewidth': 0.3,
                'linewidth': 0.0,  # i.e. without edge
            },
            'fontsize': 'small',  # or 'medium'
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'zorder': 0,
        }
        for cell_nb in range(0, nb_cells):
            text = "c{:02d}".format(cell_nb)
            xy = (x[cell_nb], y[cell_nb])
            xytext = (0.0, 5.0)
            ax.annotate(
                text, xy, xytext=xytext, xycoords='data', textcoords='offset points', **text_kwargs
            )
        # Add identity line.
        ax.plot([0.0, 1.0], [0.0, 1.0], color='grey', linewidth=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.xaxis.set_major_locator(plt.MaxNLocator(n=2))
        # ax.yaxis.set_major_locator(plt.MaxNLocator(n=2))
        # ax.xaxis.set_major_locator(plt.LinearLocator(numticks=2))
        # ax.yaxis.set_major_locator(plt.LinearLocator(numticks=2))
        ax.xaxis.set_major_locator(plt.FixedLocator([0.0, 1.0]))
        ax.yaxis.set_major_locator(plt.FixedLocator([0.0, 1.0]))
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel("reliability")
        ax.set_ylabel("accuracy")
        fig.tight_layout()
        # Add inset.
        axins = ax.inset_axes([0.03, 0.53, 0.47, 0.47])
        if highlighted_cells is None:
            axins.scatter(x, y, c='black', **scatter_kwargs)
        else:
            axins.scatter(x, y, c='black', **scatter_kwargs)
            axins.scatter(x[highlighted_cells], y[highlighted_cells], c='red', **scatter_kwargs)
            # TODO find the proper way to do that!
        for cell_nb in range(0, nb_cells):
            text = "c{:02d}".format(cell_nb)
            xy = (x[cell_nb], y[cell_nb])
            xytext = (0.0, 5.0)
            axins.annotate(
                text, xy, xytext=xytext, xycoords='data', textcoords='offset points', **text_kwargs
            )
        # Add identity line.
        axins.plot([0.0, 1.0], [0.0, 1.0], color='grey', linewidth=0.3)
        axins.set_xlim(0.8, 1.0)
        axins.set_ylim(0.8, 1.0)
        axins.xaxis.set_major_locator(plt.FixedLocator([0.8, 1.0]))
        axins.yaxis.set_major_locator(plt.FixedLocator([0.8, 1.0]))
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        ax.indicate_inset_zoom(axins)
        # fig.tight_layout()

        # Save plot (if necessary).
        plot_path = self._get_plot_path()
        if plot_path is not None:
            suffix = "_{}".format(self._tag) if self._tag is not None else ""
            output_filename = "evaluation_predictions_populationwise{}.pdf".format(suffix)
            output_path = os.path.join(plot_path, output_filename)
            fig.savefig(output_path)
            plt.close(fig)

        # Plot accuracies vs accuracies from other models.
        plot_path = self._get_plot_path()
        # Load accuracies for the LN model.
        ln_accuracies = None
        if plot_path is not None:
            input_path = os.path.join(plot_path, "evaluation_predictions_ln_accuracies.npy")
            if os.path.isfile(input_path):
                ln_accuracies = np.load(input_path)
        # Load accuracies for the LN-LN model.
        lnln_accuracies = None
        if plot_path is not None:
            input_path = os.path.join(plot_path, "evaluation_predictions_lnln_accuracies.npy")
            if os.path.isfile(input_path):
                lnln_accuracies = np.load(input_path)
        # Plot accuracies vs LN accuracies (if possible).
        if ln_accuracies is not None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            x = ln_accuracies
            y = accuracies
            c = ['black' for _ in range(0, nb_cells)]
            if highlighted_cells is not None:
                for cell_nb in highlighted_cells:
                    c[cell_nb] = 'red'
            scatter_kwargs = {
                's': 2 ** 2,
            }
            ax.scatter(x, y, c=c, **scatter_kwargs)
            # # Add point/cell labels.
            text_kwargs = {
                'color': 'grey',
                'bbox': {
                    'boxstyle': 'round',
                    'alpha': 0.3,
                    # 'edgecolor': 'black',
                    'facecolor': 'white',
                    # 'linewidth': 0.3,
                    'linewidth': 0.0,  # i.e. without edge
                },
                'fontsize': 'small',  # or 'medium'
                'horizontalalignment': 'center',
                'verticalalignment': 'center',
                'zorder': 0,
            }
            for cell_nb in range(0, nb_cells):
                text = "c{:02d}".format(cell_nb)
                xy = (x[cell_nb], y[cell_nb])
                xytext = (0.0, 5.0)
                ax.annotate(
                    text, xy, xytext=xytext, xycoords='data', textcoords='offset points', **text_kwargs
                )
            # # Add identity line.
            ax.plot([0.0, 1.0], [0.0, 1.0], color='grey', linewidth=0.3)
            # # Set axis limits.
            v_min, v_max = 0.0, 1.0
            x_min = v_min - margin_factor * (v_max - v_min)
            x_max = v_max + margin_factor * (v_max - v_min)
            y_min = v_min - margin_factor * (v_max - v_min)
            y_max = v_max + margin_factor * (v_max - v_min)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            # # Set axis tick locations.
            ax.xaxis.set_major_locator(plt.FixedLocator([0.0, 1.0]))
            ax.yaxis.set_major_locator(plt.FixedLocator([0.0, 1.0]))
            # # Hide the right and top spines.
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # # Set axis labels.
            ax.set_xlabel("LN accuracy")
            ax.set_ylabel("accuracy")
            fig.tight_layout()
            # Save plot (if necessary).
            plot_path = self._get_plot_path()
            if plot_path is not None:
                suffix = "_{}".format(self._tag) if self._tag is not None else ""
                output_filename = "evaluation_predictions_accuracies_vs_ln_accuracies{}.pdf".format(suffix)
                output_path = os.path.join(plot_path, output_filename)
                fig.savefig(output_path)
                plt.close(fig)
        # Plot accuracies vs LN-LN accuracies (if possible).
        if lnln_accuracies is not None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            x = lnln_accuracies
            y = accuracies
            c = ['black' for _ in range(0, nb_cells)]
            if highlighted_cells is not None:
                for cell_nb in highlighted_cells:
                    c[cell_nb] = 'red'
            scatter_kwargs = {
                's': 2 ** 2,
            }
            ax.scatter(x, y, c=c, **scatter_kwargs)
            # # Add point/cell labels.
            text_kwargs = {
                'color': 'grey',
                'bbox': {
                    'boxstyle': 'round',
                    'alpha': 0.3,
                    # 'edgecolor': 'black',
                    'facecolor': 'white',
                    # 'linewidth': 0.3,
                    'linewidth': 0.0,  # i.e. without edge
                },
                'fontsize': 'small',  # or 'medium'
                'horizontalalignment': 'center',
                'verticalalignment': 'center',
                'zorder': 0,
            }
            for cell_nb in range(0, nb_cells):
                text = "c{:02d}".format(cell_nb)
                xy = (x[cell_nb], y[cell_nb])
                xytext = (0.0, 5.0)
                ax.annotate(
                    text, xy, xytext=xytext, xycoords='data', textcoords='offset points', **text_kwargs
                )
            # # Add identity line.
            ax.plot([0.0, 1.0], [0.0, 1.0], color='grey', linewidth=0.3)
            # # Set axis limits.
            v_min, v_max = 0.0, 1.0
            x_min = v_min - margin_factor * (v_max - v_min)
            x_max = v_max + margin_factor * (v_max - v_min)
            y_min = v_min - margin_factor * (v_max - v_min)
            y_max = v_max + margin_factor * (v_max - v_min)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            # # Set axis tick locations.
            ax.xaxis.set_major_locator(plt.FixedLocator([0.0, 1.0]))
            ax.yaxis.set_major_locator(plt.FixedLocator([0.0, 1.0]))
            # # Hide the right and top spines.
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # # Set axis labels.
            ax.set_xlabel("LN-LN accuracy")
            ax.set_ylabel("accuracy")
            fig.tight_layout()
            # Save plot (if necessary).
            plot_path = self._get_plot_path()
            if plot_path is not None:
                suffix = "_{}".format(self._tag) if self._tag is not None else ""
                output_filename = "evaluation_predictions_accuracies_vs_lnln_accuracies{}.pdf".format(suffix)
                output_path = os.path.join(plot_path, output_filename)
                fig.savefig(output_path)
                plt.close(fig)

        return

    def plot_randomized_search_summary(self, nb_runs, val_data=None, test_data=None, train_data=None):

        import re

        # Collect data
        df = self.get_randomized_search_table(nb_runs)
        hyperparameter_names = [
            name
            for name in df.columns
            if re.search("^hyperparameters/", name)
        ]
        nb_hyperparameters = len(hyperparameter_names)
        assert nb_hyperparameters > 0, nb_hyperparameters
        val_poissons = df['val_poisson'].to_numpy()
        # test_accuracies = df['test_accuracy'].to_numpy()  # TODO uncomment?
        if val_data is not None:
            _, val_y_true = val_data
            nb_conditions, nb_cells = val_y_true.shape
            # Compute validation poisson for the "mean" model.
            val_y_pred = np.mean(val_y_true, axis=(0,))
            val_y_pred = np.tile(val_y_pred, (nb_conditions, 1))
            val_y_pred = np.maximum(val_y_pred, 1e-8)
            assert val_y_pred.shape == val_y_true.shape, (val_y_pred.shape, val_y_true.shape)
            mean_val_poisson = np.mean(val_y_pred - val_y_true * np.log(val_y_pred))
            # psth_val_poisson = None  # can not be computed (no repetitions)
        if train_data is not None:
            _, train_y_true = train_data
            # Compute val poisson for the "mean" model with the mean of the training.
            train_y_pred = np.mean(train_y_true, axis=(0,))
            train_y_pred = np.tile(train_y_pred, (nb_conditions, 1))
            train_y_pred = np.maximum(train_y_pred, 1e-8)
            assert train_y_pred.shape == val_y_true.shape, (train_y_pred.shape, val_y_true.shape)
            check = np.mean(train_y_pred - val_y_true * np.log(train_y_pred))
        else:
            mean_val_poisson = None
        if test_data is not None:
            _, test_y_true = test_data
            nb_repetitions, nb_conditions, nb_cells = test_y_true.shape
            # Compute test poisson for the "mean" model.
            test_y_pred = np.mean(test_y_true, axis=(0, 1)) #here I average over images and repetitions
            test_y_pred = np.tile(test_y_pred, (nb_repetitions, nb_conditions, 1))
            test_y_pred = np.maximum(test_y_pred, 1e-8)
            assert test_y_pred.shape == test_y_true.shape, (test_y_pred.shape, test_y_true.shape)
            mean_test_poisson = np.mean(test_y_pred - test_y_true * np.log(test_y_pred))
            # Compute test poisson for the "PSTH" model.
            test_y_pred = np.mean(test_y_true, axis=(0,)) #here I average only over images
            test_y_pred = np.tile(test_y_pred, (nb_repetitions, 1, 1))
            test_y_pred = np.maximum(test_y_pred, 1e-8)
            assert test_y_pred.shape == test_y_true.shape, (test_y_pred.shape, test_y_true.shape)
            psth_test_poisson = np.mean(test_y_pred - test_y_true * np.log(test_y_pred))       
        else:
            mean_test_poisson = None
            psth_test_poisson = None

        nb_rows = 1
        nb_columns = nb_hyperparameters
        fig, axes = plt.subplots(
            nrows=nb_rows, ncols=nb_columns, squeeze=False,
            figsize=(float(nb_columns) * 1.5 * 1.6, float(nb_rows) * 1.5 * 1.6)
        )

        for hyperparameter_nb, hyperparameter_name in enumerate(hyperparameter_names):

            hyperparameter_values = df[hyperparameter_name].to_numpy()

            ax = axes[0, hyperparameter_nb]
            x = hyperparameter_values
            y = val_poissons
            s = 3 ** 2
            c = val_poissons
            val_poisson_ref = np.quantile(val_poissons, 0.05)
            edgecolors = [
                'tab:red' if val_poisson < val_poisson_ref else 'none'
                for val_poisson in val_poissons
            ]
            pc = ax.scatter(x, y, s=s, c=c, edgecolors=edgecolors)
            linewidth = 0.3
            linestyle = (0, (5 / linewidth, 5 / linewidth))  # i.e. dashed (adapted)
            if mean_val_poisson is not None:
                ax.axhline(y=mean_val_poisson, color='tab:grey', linewidth=linewidth, linestyle='solid', zorder=0)
            if mean_test_poisson is not None:
                ax.axhline(y=mean_test_poisson, color='tab:grey', linewidth=linewidth, linestyle=linestyle, zorder=0)
            if check is not None:
                ax.axhline(y=check, color='red', linewidth=linewidth, linestyle='solid', zorder=0)
            #if psth_test_poisson is not None:
                #ax.axhline(y=psth_test_poisson, color='tab:grey', linewidth=0.3, linestyle=linestyle, zorder=0)
            # Set axis scales.
            ax.set_xscale('log')
            ax.set_yscale('linear')
            # Set axis limits.
            ax.set_xlim(np.min(x) / 2.0, np.max(x) * 2.0)
            # Set axis tick locations.
            # ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=2))
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Set axis labels.
            # xlabel = hyperparameter_name.replace("_", "\_")
            xlabel = hyperparameter_name.replace("_", "\_").replace("hyperparameters/", "")
            ax.set_xlabel(xlabel)
            if hyperparameter_nb == 0:
                ax.set_ylabel("validation loss")
            else:
                ax.set_yticklabels([])
            # Tight layout.
            fig.tight_layout()
            # # Add colorbar.
            # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            # axins = inset_axes(ax, width="3%", height="50%", loc='upper right')
            # cb = fig.colorbar(pc, cax=axins, orientation="vertical")
            # cb.set_label("validation loss")
            # axins.yaxis.set_ticks_position('left')
            # axins.yaxis.set_label_position('left')
            # axins.yaxis.set_major_locator(plt.MaxNLocator(nbins=2))
            _ = pc

        # Save plot (if necessary).
        plot_path = self._get_plot_path()
        if plot_path is not None:
            output_path = os.path.join(plot_path, "randomized_search.pdf")
            fig.savefig(output_path)
            plt.close(fig)

        return

    def plot_model_summary(self, test_data=None):

        # ...
        assert self._run_name is not None
        tensorboard_scalars = self.get_tensorboard_scalars(self._run_name)
        tensorboard_tensors = self.get_tensorboard_tensors(self._run_name)
        # ...
        if test_data is not None:
            test_x, test_y = test_data
            # test_activation = self.predict_activation(test_data)  # TODO remove?
            test_y_pred = self.predict(test_data)
        else:
            test_x, test_y = None, None
            # test_activation = None  # TODO remove?
            test_y_pred = None

        # Create figure.
        nb_rows = 1
        nb_columns = 1
        figsize = (
            float(nb_columns) * 3.0 * 1.6,
            float(nb_rows) * 3.0 * 1.6,
        )
        fig, axes = plt.subplots(nrows=nb_rows, ncols=nb_columns, squeeze=False, figsize=figsize)

        # Plot loss & metric through epoch.
        ax = axes[0, 0]
        # # Set axis scales.
        # ax.set_xscale('linear')
        # ax.set_yscale('log')
        # Plot train loss.
        x = tensorboard_scalars["epoch_loss"]["step"]
        y = tensorboard_scalars["epoch_loss"]["value"]
        c = 'C0'
        alpha = 0.4
        ax.plot(x, y, c=c, alpha=alpha, label="train loss")
        # # Plot train poisson.
        x = tensorboard_scalars["epoch_poisson"]["step"]
        y = tensorboard_scalars["epoch_poisson"]["value"]
        c = 'C0'
        ax.plot(x, y, c=c, label="train Poisson")
        # # Plot validation loss.
        x = tensorboard_tensors["val_loss"]["step"]
        y = tensorboard_tensors["val_loss"]["value"]
        c = 'C1'
        alpha = 0.4
        ax.plot(x, y, c=c, alpha=alpha, label="val. loss")
        # # Plot validation Poisson.
        x = tensorboard_tensors["val_poisson"]["step"]
        y = tensorboard_tensors["val_poisson"]["value"]
        c = 'C1'
        ax.plot(x, y, c=c, label="val. Poisson")
        # # Set axis limits.
        # ax.set_xlim(x[0], x[-1])
        # # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # # Set axis locators.
        ax.xaxis.set_major_locator(plt.FixedLocator([x[0], x[-1]]))
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        # # Set axis labels.
        ax.set_xlabel("epoch")
        # ax.set_ylabel("value")
        # # Set title.
        ax.set_title(r"losses \& metrics")
        # # Add annotation.
        final_val_poisson = tensorboard_tensors["val_poisson"]["value"][-1]
        ax.annotate(
            "final val. Poisson\n" + r"$\simeq " + "{:.3f}".format(final_val_poisson) + "$",
            # xy=(0.0, 0.0),
            # xytext=(0.0, -50.0),
            xy=(0.75, 0.25),
            xytext=(0.0, 0.0),
            xycoords='axes fraction',
            textcoords='offset points',
            # horizontalalignment='left',
            horizontalalignment='center',
        )
        # # Add legend.
        ax.legend()

        # TODO complete?

        # Tight layout.
        fig.tight_layout()

        # # Add annotations.
        # ax = axes[0, 0]
        # if self._name is not None:
        #     text = self._name
        #     ax.annotate(
        #         text, (0.0, 1.0), (+7.0, -7.0),
        #         xycoords='figure fraction', textcoords='offset points',
        #         horizontalalignment='left', verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='tab:grey')
        #     )
        # if self._run_name is not None:
        #     text = self._run_name.replace("run_", "r")
        #     ax.annotate(
        #         text, (0.0, 1.0), (+7.0 + 25.0, -7.0),
        #         xycoords='figure fraction', textcoords='offset points',
        #         horizontalalignment='left', verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='tab:grey')
        #     )

        # Save plot (if necessary).
        plot_path = self._get_plot_path(with_run_name=True)
        if plot_path is not None:
            output_path = os.path.join(plot_path, "model_summary.pdf")
            fig.savefig(output_path)
            plt.close(fig)

        return

    def plot_local_spike_triggered_averages(
            self, test_data, nb_columns=10, sta_ellipse_parameters_path=None, pixel_size=None, force=False
    ):

        test_x, _ = test_data
        images = test_x[:, :, :, 0]
        nb_images, _, _ = images.shape
        lstas = self.predict_lstas(test_data)
        nb_cells, _, _, _ = lstas.shape

        for cell_nb in range(0, nb_cells):

            plot_path = self._get_plot_path(with_run_name=True)
            if plot_path is not None:
                output_directory = os.path.join(plot_path, "local_stas")
                if not os.path.isdir(output_directory):
                    os.makedirs(output_directory)
                prefix = "c{:02d}_".format(cell_nb)
                prefix = prefix + "{}_".format(self._tag) if self._tag is not None else prefix
                output_filename = "{}local_stas.pdf".format(prefix)
                output_path = os.path.join(output_directory, output_filename)
            else:
                output_path = None

            if output_path is not None and os.path.isfile(output_path) and not force:
                continue

            cell_lstas = lstas[cell_nb]

            # Load STA ellipse (if possible).
            if sta_ellipse_parameters_path is not None and cell_nb is not None and pixel_size is not None:
                # Load ellipse.
                df = pd.read_csv(sta_ellipse_parameters_path, index_col=0)
                parameters = df.loc[cell_nb]
                xy = (parameters['x'], parameters['y'])
                w = parameters['w']
                h = parameters['h']
                a = parameters['a']
                ellipse = pcs.Ellipse(xy, w, h, angle=a, fill=False, color='tab:green', alpha=0.5)
                # Set image extent.
                _, width, height = images.shape
                left = -0.5 * float(width) * pixel_size
                right = +0.5 * float(width) * pixel_size
                bottom = -0.5 * float(height) * pixel_size
                top = +0.5 * float(height) * pixel_size
                extent = (left, right, bottom, top)
                # Set axis limits.
                zoom_window = 1.0e-3
                zoom_window = min(zoom_window, right - left, top - bottom)
                x_0, y_0 = xy
                x_0 = max(x_0, left + 0.5 * zoom_window)
                x_0 = min(x_0, right - 0.5 * zoom_window)
                y_0 = max(y_0, bottom + 0.5 * zoom_window)
                y_0 = min(y_0, top - 0.5 * zoom_window)
                x_limits = (x_0 - 0.5 * zoom_window, x_0 + 0.5 * zoom_window)
                y_limits = (y_0 - 0.5 * zoom_window, y_0 + 0.5 * zoom_window)
            else:
                ellipse = None
                extent = None
                x_limits = None
                y_limits = None

            # Create figure.
            nb_rows = 2 * ((nb_images - 1) // nb_columns + 1)
            # figsize = (
            #     float(nb_columns) * 1.0 * 1.6,
            #     float(nb_rows) * 1.0 * 1.6,
            # )
            # figsize = (6.4, 4.8)  # i.e. Matplotlib's default
            # figsize = (5.0, 7.5)  # i.e. LaTeX's default (in in)
            # figsize = (5.0, 3.0)  # good
            # figsize = (5.0, 2.5)  # bad
            figsize = (
                5.0,
                float(nb_rows) * 5.0 / float(nb_columns)
            )
            gridspec_kwargs = {
                'left': 0.02,  # (fraction of figure width)
                'right': 0.98,
                'top': 0.98,  # (fraction of figure height)
                'bottom': 0.02,
                'wspace': 0.05,  # (fraction of average axis width)
                'hspace': 0.05,  # (fraction of average axis height)
            }
            fig, axes = plt.subplots(
                nrows=nb_rows, ncols=nb_columns, squeeze=False, gridspec_kw=gridspec_kwargs, figsize=figsize
            )

            # Hide axis for each subplot.
            for ax in axes.flat:
                ax.set_axis_off()

            # image_imshow_kwargs = {
            #     'cmap': 'RdBu_r',
            #     'vmin': - 0.25 * np.max(np.abs(images)),
            #     'vmax': + 0.25 * np.max(np.abs(images)),
            #     'extent': extent,
            # }
            image_imshow_kwargs = {
                'cmap': 'Greys_r',
                'vmin': np.min(images),
                'vmax': np.max(images),
                'extent': extent,
            }

            # lsta_imshow_kwargs = {
            #     'cmap': 'RdBu_r',
            #     'vmin': - np.max(np.abs(lstas)),
            #     'vmax': + np.max(np.abs(lstas)),
            #     'extent': extent,
            # }
            lsta_imshow_kwargs = {
                'cmap': 'RdBu_r',
                'vmin': - np.max(np.abs(cell_lstas)),
                'vmax': + np.max(np.abs(cell_lstas)),
                'extent': extent,
            }

            for k in range(0, nb_images):

                # Plot image.
                image = images[k]
                row_nb = 2 * (k // nb_columns)
                column_nb = k % nb_columns
                ax = axes[row_nb, column_nb]
                ax.set_axis_on()
                ax.imshow(image, **image_imshow_kwargs)
                ax.set_xticks([])
                ax.set_yticks([])
                if ellipse is not None:
                    ellipse = pcs.Ellipse(xy, w, h, angle=a, fill=False, color='tab:green', alpha=0.5)
                    ax.add_patch(ellipse)
                    ax.set_xlim(*x_limits)  # i.e. zoom
                    ax.set_ylim(*y_limits)  # i.e. zoom
                    plot_scale_bar(ax, scale=250.0e-6, unit='m', loc='lower left', with_label=False, color='white')

                # Plot LSTA.
                lsta = cell_lstas[k]
                row_nb = 2 * (k // nb_columns) + 1
                column_nb = k % nb_columns
                ax = axes[row_nb, column_nb]
                ax.set_axis_on()
                ax.imshow(lsta, **lsta_imshow_kwargs)
                ax.set_xticks([])
                ax.set_yticks([])
                if ellipse is not None:
                    ellipse = pcs.Ellipse(xy, w, h, angle=a, fill=False, color='tab:green', alpha=0.5)
                    ax.add_patch(ellipse)
                    ax.set_xlim(*x_limits)  # i.e. zoom
                    ax.set_ylim(*y_limits)  # i.e. zoom
                    plot_scale_bar(ax, scale=250e-6, unit='m', loc='lower left', with_label=(k == 0))

            # # Tight layout.
            # fig.tight_layout()

            # Add annotations.
            ax = axes[0, 0]
            text = "c{:02d}".format(cell_nb)
            ax.annotate(
                text, (0.0, 1.0), (+7.0, -7.0),
                xycoords='figure fraction', textcoords='offset points',
                horizontalalignment='left', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='tab:grey')
            )
            if self._run_name is not None:
                text = self._run_name.replace("run_", "r")
                ax.annotate(
                    text, (0.0, 1.0), (+7.0 + 25.0, -7.0),
                    xycoords='figure fraction', textcoords='offset points',
                    horizontalalignment='left', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='tab:grey')
                )

            # Save plot (if necessary).
            if output_path is not None:
                fig.savefig(output_path)
                plt.close(fig)

        return

    plot_local_stas = plot_local_spike_triggered_averages  # i.e. alias
    plot_lstas = plot_local_spike_triggered_averages  # i.e. alias
