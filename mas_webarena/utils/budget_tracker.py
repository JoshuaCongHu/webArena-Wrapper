import time

class BudgetTracker:
    """Track budget consumption during task execution"""
    
    def __init__(self, initial_budget: float = 1.0):
        self.initial_budget = initial_budget
        self.remaining = initial_budget
        self.spent = 0.0
        self.history = []
        
    def consume(self, amount: float) -> bool:
        """Consume budget, return False if insufficient"""
        if amount > self.remaining:
            return False
        self.remaining -= amount
        self.spent += amount
        self.history.append({
            'amount': amount,
            'remaining': self.remaining,
            'timestamp': time.time()
        })
        return True
        
    def reset(self, budget: float = None):
        """Reset budget for new task"""
        self.initial_budget = budget or self.initial_budget
        self.remaining = self.initial_budget
        self.spent = 0.0
        self.history = []