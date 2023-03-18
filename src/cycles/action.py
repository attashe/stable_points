


class Action:
    
    def __init__(self) -> None:
        pass
    
    def __call__(self):
        raise Exception('Not implemented error')
    

class ActionBuilder(object):
    
    def __init__(self) -> None:
        super().__init__()
        
    def create_action(self, func, *args, **kwargs):
        pass