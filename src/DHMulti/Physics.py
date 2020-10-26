from src.DH.Physics import DHPhysics, DHPhysicsParams
from src.ModelStats import ModelStats

class DHMultiPhysics(DHPhysics):

    def __init__(self, params: DHPhysicsParams, stats: ModelStats):
        super().__init__(params, stats)

    def get_movement_budget_used(self):
        return sum(self.state.initial_movement_budgets) - sum(self.state.movement_budgets)

    def get_cral(self):
        return self.get_collection_ratio() * self.state.all_landed

    def get_movement_ratio(self):
        return float(self.get_movement_budget_used()) / float(sum(self.state.initial_movement_budgets))

    def has_landed(self):
        return self.state.all_landed
