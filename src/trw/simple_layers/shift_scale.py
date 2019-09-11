from trw.simple_layers import simple_layers
import trw


class ShiftScale(simple_layers.SimpleModule):
    """
    Normalize a tensor with a mean and standard deviation

    The output tensor will be (x - mean) / standard_deviation

    This layer simplify the preprocessing for the `trw.simple_layers` package
    """
    def __init__(self, node, mean, standard_deviation):
        """
        Args:
            node: the input node
            mean: the mean. Must be broadcastable to node shape
            standard_deviation: the standard deviation. Must be broadcastable to node shape
        """
        super().__init__(node=node, module=trw.layers.ShiftScale(mean, standard_deviation), shape=node.shape)


