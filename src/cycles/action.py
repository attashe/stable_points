


class Action:
    
    def __init__(self, func):
        self.function = func
        
    def run(self, state, params):
        return self.function(state, params)
    

class ActionBuilder(object):
    
    def __init__(self) -> None:
        super().__init__()
        
    def create_action(self, func, *args, **kwargs):
        pass
