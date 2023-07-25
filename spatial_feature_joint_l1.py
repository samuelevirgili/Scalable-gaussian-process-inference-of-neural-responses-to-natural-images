import logging
import tensorflow as tf

from pyretina_systemidentification.initializers.sta import KlindtSTAInitializer
from pyretina_systemidentification.initializers.sta import EckerSTAInitializer


logger = logging.getLogger(__name__)


class SpatialXFeatureJointL1Readout(tf.keras.layers.Layer):

    def __init__(
            self, nb_cells=1, x=None, y=None, spatial_masks_initializer='truncated normal',  # TODO correct!
            feature_weights_initializer=None,  # TODO correct?
            non_negative_feature_weights=False, spatial_sparsity_factor=0.01, feature_sparsity_factor=0.01,
            name='spatial_x_feature_joint_l1_readout', **kwargs
    ):
        """Input-independent initialization of the layer."""
        super().__init__(name=name, **kwargs)
        # ...
        self.nb_cells = nb_cells
        self.x = x
        self.y = y
        self.spatial_masks_initializer = spatial_masks_initializer
        self.feature_weights_initializer = feature_weights_initializer
        self.non_negative_feature_weights = non_negative_feature_weights
        self.spatial_sparsity_factor = spatial_sparsity_factor
        self.feature_sparsity_factor = feature_sparsity_factor
        # ...
        self.masks = None
        self.feature_weights = None
        self.biases = None
        # ...
        self.losses_map = dict()

    def build(self, input_shape):
        """Input-dependent initialization of the layer."""
        nb_samples, nb_vertical_pixels, nb_horizontal_pixels, nb_features = input_shape
        # Initialize spatial masks.
        if self.spatial_masks_initializer == 'truncated normal':
            logger.warning("spatial masks initialization may generate negative values")  # TODO improve!
            masks_initializer = tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                stddev=0.01,  # TODO use 0.05 instead (default value)?
                # seed=None,
            )
        elif self.spatial_masks_initializer == '[Klindt et al., 2017]':
            masks_initializer = KlindtSTAInitializer(
                self.x,
                self.y,
                # mean=0.0,
                stddev=0.001,
                # seed=None,
            )
        elif self.spatial_masks_initializer == '[Ecker et al., 2019]':
            logger.warning("spatial masks initialization may generate negative values")  # TODO improve!
            masks_initializer = EckerSTAInitializer(
                self.x,
                self.y,
                # mean=0.0,
                stddev=0.001,
                # seed=None,
            )
        else:
            masks_initializer = self.spatial_masks_initializer
        masks_regularizer = tf.keras.regularizers.L1L2(
            l1=self.spatial_sparsity_factor,
            l2=0.0
        )
        self.masks = self.add_weight(
            name='masks',
            shape=(self.nb_cells, nb_vertical_pixels, nb_horizontal_pixels),
            dtype=tf.float32,
            initializer=masks_initializer,
            regularizer=masks_regularizer,
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
            # partitioner=None,
            # use_resource=None,
            # synchronization=tf.VariableSynchronization.AUTO,
            # aggregation=tf.compat.v1.VariableAggregation.NONE,
            # **kwargs,
        )
        # Initialize feature weights.
        if self.feature_weights_initializer == 'truncated normal':
            if self.non_negative_feature_weights:
                logger.warning("feature weights initialization may generate negative values")  # TODO improve!
            feature_weights_initializer = tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                stddev=0.05,
            )
        elif self.feature_weights_initializer == '[Klindt et al., 2017]':
            mean = 1.0 / float(nb_features)
            stddev = 0.01
            if self.non_negative_feature_weights:
                if mean - 2.0 * stddev < 0.0:
                    logger.warning("feature weights initialization may generate negative values")  # TODO improve!
            feature_weights_initializer = tf.keras.initializers.TruncatedNormal(
                mean=mean,
                stddev=stddev,
            )
        elif self.feature_weights_initializer == '[Ecker et al., 2019]':
            if self.non_negative_feature_weights:
                logger.warning("feature weights initialization may generate negative values")  # TODO improve!
            feature_weights_initializer = tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                stddev=0.01,
            )
        else:
            feature_weights_initializer = self.feature_weights_initializer
        feature_weights_regularizer = tf.keras.regularizers.L1L2(
            l1=self.feature_sparsity_factor,
            l2=0.0,
        )
        self.feature_weights = self.add_weight(
            name='feature_weights',
            shape=(self.nb_cells, nb_features),
            dtype=tf.float32,
            initializer=feature_weights_initializer,
            regularizer=feature_weights_regularizer,
            trainable=True,
            constraint=tf.keras.constraints.NonNeg() if self.non_negative_feature_weights else None,
            # partitioner=None,
            # use_resource=None,
            # synchronization=tf.VariableSynchronization.AUTO,
            # aggregation=tf.compat.v1.VariableAggregation.NONE,
            # **kwargs
        )
        # Initialize the biases.
        biases_initializer = tf.initializers.constant(
            value=0.5
        )
        # TODO use `self.add_weight` instead of `tf.Variable`.
        self.biases = tf.Variable(
            initial_value=biases_initializer(
                shape=(self.nb_cells,),
                dtype=tf.float32,
            ),
            trainable=True,
            name='biases',
        )

        return

    def call(self, inputs, **kwargs):
        """Forward computation of the layer."""

        # Implement "mask".
        axes = [
            [1, 2],
            [1, 2],
        ]
        masked = tf.tensordot(inputs, self.masks, axes)  # i.e. tensor contraction
        # Implement "feature weights".
        h = tf.math.reduce_sum(masked * tf.transpose(self.feature_weights), axis=1)
        # Output non-linearity.
        x = tf.identity(tf.math.softplus(h + self.biases), name='output')  # TODO enable!
        outputs = x

        # Register loss terms (i.e. regularization).
        # logger.debug("dir(self.masks): {}".format(dir(self.masks)))
        logger.debug("self.masks.name: {}".format(self.masks.name))
        # logger.debug("dir(self.feature_weights): {}".format(dir(self.feature_weights)))
        logger.debug("self.feature_weights.name: {}".format(self.feature_weights.name))
        logger.debug("self.losses: {}".format(self.losses))
        assert len(self.losses) == 2, self.losses
        # TODO how to find an explicit correspondence between weights and corresponding losses (regularization).
        # TODO (something which also works in TF eager mode!)
        weights = [self.masks, self.feature_weights]  # this weight order should match the loss order
        for loss_nb, weight in enumerate(weights):
            loss_name = weight.name
            loss_name = loss_name[loss_name.find('/')+1:]  # i.e. remove model name
            loss_name = loss_name[:loss_name.find(':')]  # i.e. remove trailing index
            loss_name = "regularization/{}".format(loss_name)  # i.e. add prefix
            loss_value = self.losses[loss_nb]
            self.losses_map[loss_name] = loss_value

        # Add regularization terms as metrics.
        # logger.debug("self.losses: {}".format(self.losses))
        logger.debug("self.losses_map: {}".format(self.losses_map))
        for loss_name, loss_value in self.losses_map.items():
            self.add_metric(loss_value, aggregation='mean', name=loss_name)

        return outputs

    # def get_config(self):

    # @classmethod
    # def from_config(cls, config):
