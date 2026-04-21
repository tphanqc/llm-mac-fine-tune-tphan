class ModelRegistry:
    _models = {}

    @classmethod
    def register(cls, name):
        def wrapper(wrapped_class):
            cls._models[name] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def get_loader(cls, name):
        if name not in cls._models:
            raise ValueError(f"Model loader '{name}' not found in registry.")
        return cls._models[name]

registry = ModelRegistry()
