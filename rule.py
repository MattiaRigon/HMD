class Rule:
    """Base class for validation rules."""
    def validate(self, value):
        raise NotImplementedError("Subclasses must implement this method.")

class InListRuleFromString(Rule):
    """Rule to validate if a value is in a predefined list."""
    def __init__(self, allowed_values):
        self.allowed_values = allowed_values

    def validate(self, value):
        for val in self.allowed_values.split(","):
            if val in value:
                return True

class InListRule(Rule):
    """Rule to validate if a value is in a predefined list."""
    def __init__(self, allowed_values):
        self.allowed_values = allowed_values

    def validate(self, value):
        return value in self.allowed_values

class AlwaysTrueRule(Rule):
    """Rule that always evaluates to True."""
    def validate(self, value):
        return True

class IsIntegerRule(Rule):
    """Rule to validate if a value is an integer."""
    def validate(self, value):
        try:
            x = int(value)
            return True
        except ValueError:
            return False
    
class IsStringRule(Rule):
    """Rule to validate if a value is a string."""
    def validate(self, value):
        return isinstance(value, str)

class RangeRule(Rule):
    """Rule to validate if a value is within a range."""
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value):
        return isinstance(value, int) and self.min_value <= value <= self.max_value
