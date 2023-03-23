from typing import List

from action import Action, ActionBuilder


class ScenarioRunner:
    """Create scenario with interpolation between actions:
    
    Action_0 | Interpolated_0, _1, ... interpolation_steps_0 | Action_1 | Interp....steps_1 | ... | Action_end
    """
    def __init__(self, actions = [], interpolations = []) -> None:
        self.actions: List[Action] = actions
        self.interpolations: List[int] = interpolations
    
    def add_action(self, action, interpolation_steps = None):
        if interpolation_steps is None and len(self.actions) > 0:
            raise Exception('Only first action could be without interpolation')
        
        if len(self.actions) != 0:
            self.interpolations.append(interpolation_steps)

        self.actions.append(action)
        
    def interpolate_generator(self, prev_action: Action, cur_action: Action,
                              prev_values, cur_values, steps):
        
        actions = [ActionBuilder.create_action(prev_action) for i in range(steps)]
        
        values = self._values_interpolator(prev_values, cur_values, steps)
        
        for i in range(steps):
            pass
            
    def run(self, init_values: dict):
        
        prev = None
        results = None
        for action in self.actions:
            if prev is None:
                results = action.run(init_values)
                prev = action
                continue
            
            for interp_action in self.interpolate_generator():
                results = interp_action.run(results)

            prev = action
