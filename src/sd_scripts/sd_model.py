

def load_model(model_path, config_path):
    pass


class SDModel:
    
    def __init__(self, model_path: str, config_path: str = None, version=1):
        model, config = load_model(model_path, config_path)
