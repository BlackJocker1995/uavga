class ParameterManager:
    """Manage vehicle parameters"""
    
    def __init__(self):
        self.params = {}
        self.validators = {}
        
    def register_param(self, name, validator=None, default=None):
        """Register parameter with validation"""
        self.params[name] = default
        if validator:
            self.validators[name] = validator
            
    def set_param(self, name, value):
        """Set parameter with validation"""
        if name in self.validators:
            if not self.validators[name](value):
                raise ValueError(f"Invalid value for {name}")
        self.params[name] = value
