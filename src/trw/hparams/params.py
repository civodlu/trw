import numpy as np


class HyperParam:
    pass


class DiscreteMapping(HyperParam):
    """
    Map discrete value to another discrete value
    
    e.g., this can be useful to test activation function as hyper-parameter
    """
    def __init__(self, list_name_value, current_value):
        assert isinstance(list_name_value, list)
        assert isinstance(list_name_value[0], (tuple, list))
        self.kvp = {k: v for k, v in list_name_value}
        self.list_name_value = list_name_value
        assert len(self.kvp) == len(list_name_value), 'must be the same size. duplicate name?'

        assert self.kvp.get(current_value) is not None, 'current_value must be in the dictionary! c=%s' % \
                                                        str(current_value)
        self.current_value = None
        self.set_value(current_value)

    def set_value(self, value):
        assert self.kvp.get(value) is not None, 'current_value must be in the dictionary! c=%s' % \
                                                        str(value)
        self.current_value = value

    def get_value(self):
        return self.kvp[self.current_value]

    def random_value(self):
        """
        :return: a random value
        """
        v = np.random.randint(low=0, high=len(self.kvp))
        return self.list_name_value[v][0]

    def __repr__(self):
        return 'DiscreteMapping value=%s' % str(self.current_value)


class DiscreteValue(HyperParam):
    """
    Discrete value. This can be useful to select one choice among many
    """
    def __init__(self, values, current_value):
        assert isinstance(values, list)
        assert current_value in values
        self.values = values
        self.current_value = current_value

    def set_value(self, value):
        assert value in self.values
        self.current_value = value

    def get_value(self):
        return self.current_value

    def random_value(self):
        """
        :return: a random value
        """
        v = np.random.randint(low=0, high=len(self.values))
        return self.values[v]

    def __repr__(self):
        return 'DiscreteValue value=%s' % str(self.current_value)


class DiscreteIntegrer(HyperParam):
    """
    Represent an integer hyper-parameter
    """
    def __init__(self, current_value, min_range, max_range):
        """
        :param name: the name of the hyper-parameter. Must be unique
        :param max_range: max integer (inclusive) to be generated
        :param min_range: minimum integer (inclusive) to be generated
        :param current_value:
        """
        assert max_range >= min_range
        self.max_range = max_range
        self.min_range = min_range
        self.current_value = current_value

    def set_value(self, value):
        self.current_value = value

    def get_value(self):
        return int(self.current_value)

    def random_value(self):
        """
        :return: a random value
        """
        return np.random.randint(low=self.min_range, high=self.max_range + 1)

    def __repr__(self):
        return 'DiscreteIntegrer value=%d, min=%d, max=%d' % (self.current_value, self.min_range, self.max_range)


class DiscreteBoolean(HyperParam):
    """
    Represent a boolean hyper-parameter
    """
    def __init__(self, current_value):
        """
        :param name: the name of the hyper-parameter. Must be unique
        :param current_value: the initial boolean value
        """
        assert current_value or not current_value
        self.current_value = current_value

    def set_value(self, value):
        self.current_value = value

    def get_value(self):
        return self.current_value

    def random_value(self):
        """
        :return: a random value
        """
        return np.random.randint(low=0, high=1 + 1)

    def __repr__(self):
        return 'DiscreteBoolean value=%d' % self.current_value


class ContinuousUniform(HyperParam):
    """
    Represent a continuous hyper-parameter
    """
    def __init__(self, current_value, min_range, max_range):
        """
        :param name: the name of the hyper-parameter. Must be unique
        :param max_range: max integer (inclusive) to be generated
        :param min_range: minimum integer (inclusive) to be generated
        :param current_value:
        """
        assert max_range >= min_range
        self.max_range = max_range
        self.min_range = min_range
        self.current_value = current_value

    def set_value(self, value):
        self.current_value = value

    def get_value(self):
        return self.current_value

    def random_value(self):
        """
        :return: a random value
        """
        return np.random.uniform(low=self.min_range, high=self.max_range)

    def __repr__(self):
        return 'ContinuousUniform value=%f, min=%f, max=%f' % (self.current_value, self.min_range, self.max_range)
    
    
class ContinuousPower(HyperParam):
    """
    Represent a continuous power hyper-parameter
    
    This type of distribution can be useful to test e.g., learning rate hyper-parameter. Given a
    random number x generated from uniform interval (min_range, max_range), return 10 ** x
    """
    def __init__(self, current_value, exponent_min, exponent_max):
        """
        
        Args:
            current_value: the current value of the parameter (power will ``NOT`` be applied)
            exponent_min: minimum floating number (inclusive) of the power exponent to be generated
            exponent_max: max_range: max floating number (inclusive) of the power exponent to be generated
        """
        assert exponent_max >= exponent_min
        assert current_value >= 10 ** exponent_min, 'make sure the current value must have the power already applied and be within the generated interval'
        assert current_value <= 10 ** exponent_max, 'make sure the current value must have the power already applied and be within the generated interval'
        self.exponent_max = exponent_max
        self.exponent_min = exponent_min
        self.current_value = current_value

    def set_value(self, value):
        self.current_value = value

    def get_value(self):
        return self.current_value

    def random_value(self):
        uniform = np.random.uniform(low=self.exponent_min, high=self.exponent_max)
        return 10 ** uniform

    def __repr__(self):
        return 'ContinuousPower value=%f, min=%f, max=%f' % (self.current_value, self.exponent_min, self.exponent_max)


class HyperParameters:
    """
    Holds a repository of hyper-parameters
    """
    def __init__(self, hparams=None):
        """
        Create the hyper-parameter repository

        :param hparams: if not None, the initial parameters
        """
        if hparams is not None:
            self.hparams = hparams
        else:
            self.hparams = {}

    def create(self, hparam_name, hparam):
        """
        Create an hyper parameter if it is not already present

        :param hparam_name: the name of the hyper-parameter to create
        :param hparam: the hyper-parameter description and value
        :return: the hyper parameter value
        """
        assert isinstance(hparam, HyperParam), 'must be an instance of HyperParam'
        if self.hparams.get(hparam_name) is None:
            self.hparams[hparam_name] = hparam

        hparam_config = self.hparams[hparam_name]
        return hparam_config.get_value()

    def generate_random_hparams(self):
        """
        Set hyper-parameter to a random value
        """
        for name, hparam in self.hparams.items():
            value = hparam.random_value()
            hparam.set_value(value)

    def get_value(self, name):
        """
        Return the current value of an hyper-parameter
        """
        hparam = self.hparams.get(name)
        assert hparam is not None, 'can\'t find hparam=%s' % hparam
        return hparam.get_value()

    def __str__(self):
        return str(self.hparams)

    def __len__(self):
        return len(self.hparams)

