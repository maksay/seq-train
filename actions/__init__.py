from actions.base_action import BaseAction
from actions.score_and_nms_hypotheses_action import SimpleScoreAndNMSHypotheses
from actions.select_final_solution_action import GreedyFinalSolution

__all_actions__ = [
    "BaseAction",
    "SimpleScoreAndNMSHypotheses",
    "GreedyFinalSolution"
]


def make_action(config):
    name = config["name"] if type(config) is dict else config.name
    if name in __all_actions__:
        return globals()[name](config)
    else:
        raise Exception('The action name %s does not exist' % name)

