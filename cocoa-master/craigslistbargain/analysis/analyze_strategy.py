from __future__ import division
from __future__ import absolute_import
from builtins import object
from past.utils import old_div
import numpy as np

from . import utils

MAX_MARGIN = 2.4
MIN_MARGIN = -2.0

class StrategyAnalyzer(object):
    @classmethod
    def valid_margin(cls, margin):
        return margin <= MAX_MARGIN and margin >= MIN_MARGIN

    @classmethod
    def get_margin(cls, ex, price, agent, role, remove_outlier=True):
        agent_target = ex.scenario.kbs[agent].facts["personal"]["Target"]
        partner_target = ex.scenario.kbs[1 - agent].facts["personal"]["Target"]
        midpoint = (agent_target + partner_target) / 2.
        norm_factor = np.abs(midpoint - agent_target)
        if role == utils.SELLER:
            margin = old_div((price - midpoint), norm_factor)
        else:
            margin = old_div((midpoint - price), norm_factor)
        if remove_outlier and not cls.valid_margin(margin):
            return None
        return margin

    @classmethod
    def has_deal(cls, ex):
        if ex.outcome is None or ex.outcome['reward'] == 0 or ex.outcome.get('offer', None) is None or ex.outcome['offer']['price'] is None:
            return False
        return True

