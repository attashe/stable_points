from .interpolator import Interpolator


class Scenario:
    
    def __init__(self, actions):
        self.actions = actions
        
    def run(self, state, params):
        for action in self.actions:
            state = action.run(state, params)
        
        return state
    
    
class ScenarioRunner:
    
    def __init__(self, scenario):
        self.scenario = scenario
        
    def run(self, init_params, end_params, init_state, steps):
        params_generator = Interpolator.interp(init_params, end_params, steps)
        
        state = init_state.copy()
        for params in params_generator:
            print(f'{state=}')
            print(f'{params=}')
            state = self.scenario.run(state, params)
        
        return state