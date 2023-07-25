import logging
import tensorflow as tf

from pyretina_systemidentification.regularizers.smooth_sparse_2d import SmoothSparse2DRegularizer


logger = logging.getLogger(__name__)


class StackedConv2DCore(tf.keras.layers.Layer):

    def __init__(
        self, nbs_kernels=(16, 32), kernel_sizes=(13, 3), strides=(1, 1),
        paddings=('valid', 'valid'), dilation_rates=(1, 1), activations=('elu', 'elu'),
        smooth_factors=(0.001, None), sparse_factors=(None, 0.001),
        name='stacked_conv_2d_core', **kwargs
    ):
        """Input-independent initialization of the layer."""
        super().__init__(name=name, **kwargs)
        # ...
        self.nb_subblocks = len(nbs_kernels)  # TODO rename to `nb_layers` or `nb_sublayers` instead?
        assert len(kernel_sizes) == self.nb_subblocks, (kernel_sizes, self.nb_subblocks)
        assert len(strides) == self.nb_subblocks, (strides, self.nb_subblocks)
        assert len(paddings) == self.nb_subblocks, (paddings, self.nb_subblocks)
        assert len(dilation_rates) == self.nb_subblocks, (dilation_rates, self.nb_subblocks)
        assert len(activations) == self.nb_subblocks, (activations, self.nb_subblocks)
        assert len(smooth_factors) == self.nb_subblocks, (smooth_factors, self.nb_subblocks)
        assert len(sparse_factors) == self.nb_subblocks, (sparse_factors, self.nb_subblocks)
        self.smooth_factors = smooth_factors
        self.sparse_factors = sparse_factors
        self.losses_map = dict()
        # ...
        for subblock_nb in range(0, self.nb_subblocks):
            # ...
            kernel_initializer = tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                stddev=0.01,  # use 0.05 instead (default value)?
                # seed=None
            )
            kernel_regularizer = SmoothSparse2DRegularizer(
                smooth_factor=smooth_factors[subblock_nb],
                sparse_factor=sparse_factors[subblock_nb],
            )
            # kernel_regularizer = None
            # Add convolutional layer.
            name = 'conv_{}'.format(subblock_nb)  # TODO rename?
            # self.__convolutional_layers["{}".format(subblock_nb)] = tf.keras.layers.Convolution2D(
            setattr(
                self,
                name,
                tf.keras.layers.Convolution2D(
                    nbs_kernels[subblock_nb],
                    kernel_sizes[subblock_nb],
                    strides=strides[subblock_nb],
                    padding=paddings[subblock_nb],
                    # data_format=None,
                    dilation_rate=dilation_rates[subblock_nb],
                    activation=activations[subblock_nb],
                    # use_bias=True,
                    kernel_initializer=kernel_initializer,
                    # bias_initializer='zeros',
                    kernel_regularizer=kernel_regularizer,
                    # bias_regularizer=None,
                    # activity_regularizer=None,
                    # kernel_constraint=None,
                    # bias_constraint=None,
                    name=name,
                )
            )
            # Add batch normalization layer.
            name = 'batch_norm_{}'.format(subblock_nb)
            # self.__batch_normalization_layers["{}".format(subblock_nb)] = tf.keras.layers.BatchNormalization(
            setattr(
                self,
                name,
                tf.keras.layers.BatchNormalization(
                    # axis=-1,  # TODO check this value.
                    momentum=0.98,  # momentum for the moving average
                    # [...],
                    name=name,
                )
            )

    @property
    def convolution_layers(self):

        layers = list()
        for subblock_nb in range(0, self.nb_subblocks):
            name = 'conv_{}'.format(subblock_nb)
            layer = getattr(self, name)
            layers.append(layer)

        return layers

    # def build(self, input_shape):
    #     """Input-dependent initialization of the layer."""
    #     raise NotImplementedError  # TODO complete!

    def call(self, inputs, training=None):  # TODO `None` instead of `False` (c.f. doc.)?
        """Forward computation of the layer."""
        internals = inputs
        for subblock_nb in range(0, self.nb_subblocks):
            # internals = self.__convolutional_layers["{}".format(subblock_nb)](internals)
            # internals = self.__batch_normalization_layers["{}".format(subblock_nb)](internals, training=training)
            conv_layer = getattr(self, "conv_{}".format(subblock_nb))
            batch_norm_layer = getattr(self, "batch_norm_{}".format(subblock_nb))
            internals = conv_layer(internals)
            internals = batch_norm_layer(internals, training=training)
            # logger.debug("dir(conv_layer): {}".format(dir(conv_layer)))  # TODO remove!
            logger.debug("conv_layer.kernel.name: {}".format(conv_layer.kernel.name))  # TODO remove!
            logger.debug("conv_layer.losses: {}".format(conv_layer.losses))  # TODO remove!
            assert len(conv_layer.losses) == 1, conv_layer.losses
            loss_name = conv_layer.kernel.name
            loss_name = loss_name[loss_name.find('/')+1:]  # i.e. remove model name
            loss_name = loss_name[:loss_name.find(':')]  # i.e. remove trailing index
            loss_name = "regularization/{}".format(loss_name)  # i.e. add prefix
            loss_value = conv_layer.losses[0]  # TODO correct!
            self.losses_map[loss_name] = loss_value
            # logger.debug("dir(conv_layer.kernel): {}".format(dir(conv_layer.kernel)))  # TODO remove!
            # logger.debug("conv_layer.kernel.losses: {}".format(conv_layer.kernel.losses))  # TODO remove!

            # TODO add regularization loss.
            # TODO   get convolutional weights.
            # x = conv_layer.kernel  # TODO rename!
            # if self.smooth_factors[subblock_nb]:
            #     # lap = tf.constant([
            #     #     [+0.25, +0.50, +0.25],
            #     #     [+0.50, -3.00, +0.50],
            #     #     [+0.25, +0.50, +0.25],
            #     # ])
            #     # lap = tf.expand_dims(lap, tf.expand_dims(lap, 2), 3, name='laplacian_filter')
            #     lap = tf.constant([
            #         [+0.25, +0.50, +0.25],
            #         [+0.50, -3.00, +0.50],
            #         [+0.25, +0.50, +0.25],
            #     ], shape=(3, 3, 1, 1), name='laplacian_filter')
            #     # nb_kernels = x.get_shape().as_list()[2]
            #     # nb_kernels = x.shape[2]
            #     _, _, nb_kernels, _ = x.shape
            #     x_lap = tf.nn.depthwise_conv2d(
            #         tf.transpose(x, perm=(3, 0, 1, 2)),  # inputs
            #         tf.tile(lap, (1, 1, nb_kernels, 1)),  # filter
            #         (1, 1, 1, 1),  # strides
            #         'SAME',  # padding  # TODO check this...
            #     )
            #     smooth_regularization = math_ops.reduce_sum(
            #         math_ops.reduce_sum(math_ops.square(x_lap), axis=(1, 2, 3)) / (1e-8 + math_ops.reduce_sum(math_ops.square(x), axis=(0, 1, 2)))
            #     )
            #     smooth_regularization = tf.math.scalar_mul(
            #         self.smooth_factors[subblock_nb],
            #         smooth_regularization,
            #         name="conv_{}/smooth_regularization".format(subblock_nb),
            #     )
            #     self.add_loss(smooth_regularization)
            # if self.sparse_factors[subblock_nb]:
            #     sparse_regularization = math_ops.reduce_sum(
            #         math_ops.reduce_sum(math_ops.sqrt(math_ops.reduce_sum(math_ops.square(x), axis=(0, 1))), axis=0) / math_ops.sqrt(1e-8 + math_ops.reduce_sum(math_ops.sqrt(x), axis=(0, 1, 2)))
            #     )
            #     sparse_regularization += tf.math.scalar_mul(
            #         self.sparse_factors[subblock_nb],
            #         sparse_regularization,
            #         name="conv_{}/sparse_regularization".format(subblock_nb),
            #     )
            #     self.add_loss(sparse_regularization)
            # # regularization_name = "conv_{}/total_regularization".format(subblock_nb)
            # # regularization = tf.identity(regularization, name=regularization_name)
            # # self.add_loss(regularization)

        # Add regularization terms as metrics.
        # logger.debug("self.losses: {}".format(self.losses))
        logger.debug("self.losses_map: {}".format(self.losses_map))
        for loss_name, loss_value in self.losses_map.items():
            self.add_metric(loss_value, aggregation='mean', name=loss_name)
        # self.add_metric(regulariation, aggregation='mean', name='core_smooth_sparse_regularization')  # TODO keep? rename?
        outputs = internals
        return outputs

    # def get_config(self):
    #     raise NotImplementedError

    # def from_config(self):
    #     raise NotImplementedError
