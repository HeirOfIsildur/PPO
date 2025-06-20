import numpy as np


class CalculationsDetails():
    def __init__(self, total_secondary_units: int|None = None, sample_secondary_units: int|None = None, total_primary_units: int|None = None, sample_primary_units: int|None = None, total_zeroth_units: int|None = None, sample_zeroth_units: int|None = None):
        self.total_secondary_units = total_secondary_units
        self.sample_secondary_units = sample_secondary_units
        self.total_primary_units = total_primary_units
        self.sample_primary_units = sample_primary_units
        self.total_zeroth_units = total_zeroth_units
        self.sample_zeroth_units = sample_zeroth_units

        self.children = {}


    def initialize_with_data_points(self, data, data_points):
        self.children = {data_point: data.loc[data_point, "Cu"] for data_point in data_points}

    def get_all_primary_units(self):
        all_units = []
        for key, value in self.children.items():
            if isinstance(value, CalculationsDetails):
                all_units.extend(value.get_all_primary_units())
            else:
                all_units.append(int(key))
        return all_units

    def get_sample_total(self):
        return float(self.get_sample_mean() * self.total_secondary_units)

    def get_sample_mean(self):
        values = list(self.children.values())
        if not values:
            return None
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            return float(np.mean(values))
        else:
            return float(np.mean([v.get_sample_mean() for v in values]))

    def get_sample_variance(self):
        return (1/self.sample_primary_units) * (1/(self.sample_primary_units-1)) * sum((x.get_sample_mean() - self.get_sample_mean())**2 for x in self.children.values() if isinstance(x, CalculationsDetails))

    def get_sample_total_srswor(self):
        totals_sum = sum(x.get_sample_total() for x in self.children.values() if isinstance(x, CalculationsDetails))
        return totals_sum/self.sample_primary_units * self.total_primary_units

    def get_sample_mean_srswor(self):
        return self.get_sample_total_srswor()/ self.total_secondary_units

    def get_sample_total_variance_srswor(self):
        left_summand = self.total_primary_units * (self.total_primary_units - self.sample_primary_units) * self.get_su2() / self.sample_primary_units
        right_summand = self.total_primary_units/self.sample_primary_units*sum(x.get_partial_si2_inside_sum() for x in self.children.values())
        return left_summand + right_summand

    def get_sample_variance_srswor(self):
        return (1/(self.total_secondary_units**2)) * self.get_sample_total_variance_srswor()

    def get_su2(self):
        cluster_sample_totals = [x.get_sample_total() for x in self.children.values()]
        return self.calculate_simples_math_sample_variance(cluster_sample_totals)

    def get_partial_si2_inside_sum(self):
        return self.total_secondary_units * (self.total_secondary_units - self.sample_secondary_units) * self.calculate_simples_math_sample_variance(values = list(self.children.values())) / self.sample_secondary_units

    def calculate_simples_math_sample_variance(self, values):
        mean = float(np.mean(values))
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        n = len(values)
        return (1/(n - 1)) * squared_diff_sum

    def get_sample_total_stratified(self):
        return sum(x.get_sample_total_srswor() for x in self.children.values())

    def get_sample_mean_stratified(self):
        return self.get_sample_total_stratified()/ self.total_secondary_units

    def get_sample_total_variance_stratified(self):
        return sum(x.get_sample_total_variance_srswor() for x in self.children.values())

    def get_sample_mean_variance_stratified(self):
        return self.get_sample_total_variance_stratified() / (self.total_secondary_units**2)
