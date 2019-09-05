from unittest import TestCase
import trw
import numpy as np
import math


class TestParams(TestCase):
    def test_power_initial_value(self):
        """
        Make sure we apply the power of the initial value
        """
        p = trw.hparams.ContinuousPower(2, 0, 3)
        value = p.get_value()
        expected = 2
        assert abs(value - expected) < 1e-5
        
    def test_power_range(self):
        """
        Test the range of generated values is as expected
        """
        
        values_to_generate = 10000
        values = []
        min_power = 2
        max_power = 5
        for n in range(values_to_generate):
            p = trw.hparams.ContinuousPower(101, min_power, max_power)
            value = p.random_value()
            assert value >= 10 ** min_power
            assert value <= 10 ** max_power
            values.append(value)
            
        # check the min/max domain
        exponent_domain = np.log(np.asarray(values)) / math.log(10)
        assert np.min(exponent_domain) >= min_power
        assert np.max(exponent_domain) <= max_power
        
        # make sure we have uniform distribution of the exponent
        histogram_bins = 10
        expected_bin_count = values_to_generate / histogram_bins
        counts, intervals = np.histogram(exponent_domain, range=(min_power, max_power), bins=10)
        error_rate = np.max(np.abs(counts - expected_bin_count)) / expected_bin_count
        assert error_rate < 0.1
