class ModelRegistry:
    """
    A registry to manage translation model engines dynamically.
    Allows registering and retrieving different translation models.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        """Decorator to register a translation model class."""
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_translator(cls, name, *args, **kwargs):
        """Retrieves and instantiates a translator class by name."""
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' is not registered. Available models: {list(cls._registry.keys())}")
        return cls._registry[name](*args, **kwargs)
