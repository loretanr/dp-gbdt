# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Class to generate synthetic data."""

from collections import Counter
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import pandas as pd


class DataGenerator:
  """Class to generate synthetic data."""

  def GenerateData(self, n_samples: int) -> pd.DataFrame:
    """Generate data.

    Args:
      n_samples (int): Number of data points to generate.

    Returns:
      pd.Dataframe: Generated samples.
    """
    print('Identifying distribution of stereotype company profiles...')
    stereotypes_distribution = self.GetStereotypeDistribution(
        n_samples=n_samples)

    print('Generating {0:d} samples...'.format(n_samples))
    samples = []  # type: List[List[Union[int, float]]]

    while len(samples) < n_samples:
      stereotype = CompanyProfileStereotype()
      stereotype_features = ','.join(list(map(str, list(
          stereotype.company_features.values()))))
      if stereotype_features not in stereotypes_distribution:
        continue
      n_samples_stereotype = int(np.ceil(stereotypes_distribution[
                                   stereotype_features] * n_samples) / 100)
      samples.extend(self.GenerateSamplesFromStereotype(
          stereotype, n_samples_stereotype))

    samples = samples[:n_samples]

    header = list(CompanyProfileStereotype().company_features.keys()) + [
        'loss_probability', 'loss_cost']

    # Shuffle samples, and return a dataframe with the generated data
    print('Shuffling generated data...')
    return pd.DataFrame(data=samples, columns=header).sample(
        frac=1).reset_index(drop=True)

  @staticmethod
  def GetStereotypeDistribution(n_samples: int) -> Dict[str, float]:
    """Return the distribution of a kind of stereotype amongst n_samples.

    Args:
      n_samples (int): The number of profiles to generate to compute the
          distribution.

    Returns:
      Dict: A dictionary that maps a profile to a probability of having that
          profile.
    """
    def GetFeaturesFromProfile(
        sample_id: int) -> List[Union[int, float]]:  # pylint: disable=unused-argument
      """Return features from a company profile.

      Args:
        sample_id (int): Ignored, for parallel processing only.

      Returns:
        List: A company profile's features. e.g. [0, 0, 1, ..., 1]
      """
      return list(CompanyProfileStereotype().company_features.values())

    stereotypes = []
    with ThreadPoolExecutor(max_workers=16) as executor:
      for result in executor.map(GetFeaturesFromProfile, range(n_samples)):
        stereotypes.append(result)
    stereotypes_string = []
    for stereotype in stereotypes:
      stereotypes_string.append(','.join(list(map(str, stereotype))))
    counter = Counter(stereotypes_string)
    counts = [(count, counter[count] / len(stereotypes) * 100.) for count in
           counter]  # type: List[Tuple[str, float]]

    stereotypes_distribution = {}
    for count in counts:
      dist = np.around(count[1], decimals=2)  # type: float
      stereotypes_distribution[count[0]] = dist

    return stereotypes_distribution

  def GenerateSamplesFromStereotype(
      self,
      stereotype: 'CompanyProfileStereotype',
      n_samples: int) -> List[Any]:
    """Generate data samples that are similar to a specific company stereotype.

    Args:
      stereotype (CompanyProfileStereotype): A company profile stereotype.
      n_samples (int): How many data points from the stereotype should be
          generated.

    Returns:
      List: A list of data samples.
    """

    stereotype_samples = []  # type: List[List[Union[int, float]]]

    loss_log_normal_mean, loss_log_normal_std = self.GetLogNormalParams(
        mean_x=stereotype.loss_probability,
        std_x=((stereotype.loss_probability/100)*50))
    cost_log_normal_mean, cost_log_normal_std = self.GetLogNormalParams(
        mean_x=stereotype.potential_cost,
        std_x=((stereotype.potential_cost/100)*50))

    # Generate n_samples that follow the same company profile stereotype.
    for _ in range(n_samples):
      sample = []  # type: List[Union[int, float]]
      loss_log_normal = np.around(np.random.lognormal(
          mean=loss_log_normal_mean,
          sigma=loss_log_normal_std), decimals=3)  # type: float
      cost_log_normal = np.around(np.random.lognormal(
          mean=cost_log_normal_mean,
          sigma=cost_log_normal_std), decimals=3)  # type: float

      if loss_log_normal > 1.:
        loss_log_normal = 1.
      if loss_log_normal < 0.:
        loss_log_normal = 0.

      sample.extend(stereotype.company_features.values())
      sample.extend([loss_log_normal, cost_log_normal])
      stereotype_samples.append(sample)

    return stereotype_samples

  @staticmethod
  def GetLogNormalParams(
      mean_x: float,
      std_x: float) -> Tuple[float, float]:
    """Return mean and std parameters to create a log normal distribution.

    https://en.wikipedia.org/wiki/Log-normal_distribution

    Args:
      mean_x (float): The desired mean for the variable X.
      std_x (float): The desired std for the variable X.

    Returns:
      Tuple: The parameters mean and std to create a log normal distribution
          that respect X's mean and std.
    """
    log_normal_mean = np.log(np.square(mean_x) / np.sqrt(
        np.square(mean_x) + np.square(std_x)))
    log_normal_std = np.log(1 + (np.square(std_x) / np.square(mean_x)))
    return log_normal_mean, log_normal_std


class CompanyProfileStereotype:
  """Class that represents a company profile stereotype."""

  def __init__(self) -> None:

    # Probability of being a large or small company
    # A small company is a company with less than 2 billions revenue
    self.company_size = {
        'small': 0.70,
        'large': 0.30
    }

    # Features are potential protections / mechanisms companies have in place
    # Values are the probabilities that the attribute is True
    self.features = {
        ########################### Depends on industry size
        'company_has_pii': None,
        'company_has_pci': None,
        'company_has_phi': None,
        'company_has_credentials': None,
        'company_has_other': None,
        ###########################
        ########################### Extracted from the reports
        'company_has_recovery_plan': 0.81,
        'company_has_ids_ips': 0.89,
        'company_uses_third_party': 0.79,
        'company_has_firewalls': 0.84,
        ###########################
        ########################### Made up probabilities
        'company_trains_employees_against_social_engineering': 0.70,
        'company_has_antivirus': 0.95,
        'company_has_patching_process': 0.60,
        ###########################
    }

    self.company_type = self.DrawAttributeFromProbabilities(self.company_size)

    if self.company_type == 'small':
      # Probability of a company being in a specific sector
      # https://netdiligence.com/wp-content/uploads/2020/05/2019_NetD_Claims_
      # Study_Report_1.2.pdf
      self.probability_of_belonging_to_a_sector = {
          'hospitality': 0.03,
          'public_entity': 0.03,
          'other': 0.11,
          'nonprofit': 0.05,
          'education': 0.05,
          'technology': 0.06,
          'manufacturing': 0.08,
          'financial_services': 0.09,
          'retail': 0.09,
          'healthcare': 0.19,
          'professional_services': 0.22
      }

      # These probabilities are extracted from the Netdilligence
      # report and are probabilities of events for companies given that they
      # suffered a loss already
      self.probability_of_a_cause_knowing_loss = {
          # social engineering also includes BEC, phishing, wire transfer fraud
          'company_trains_employees_against_social_engineering': 0.52,
          'company_has_patching_process': 0.93,
          'company_has_antivirus': 0.73,
      }

      # Average cost in k$ per type of data exposed for a company
      self.cost_for_data_leaked = {
          'pci': 392,
          'phi': 259,
          'pii': 163,
          'credentials': 167,
          'other': 128
      }

      # min, mean, max breach cost per sector
      self.breach_cost_per_sector = {
          'gaming_casino': {'min': 80, 'mean': 80, 'max': 1100},
          'telecommunications': {'min': 4, 'mean': 542, 'max': 2000},
          'entertainment': {'min': 7, 'mean': 154, 'max': 764},
          'media': {'min': 5, 'mean': 328, 'max': 2500},
          'restaurant': {'min': 2, 'mean': 68, 'max': 367},
          'energy': {'min': 2, 'mean': 319, 'max': 5000},
          'transportation': {'min': 5, 'mean': 590, 'max': 17500},
          'hospitality': {'min': 6, 'mean': 260, 'max': 5700},
          'public_entity': {'min': 3, 'mean': 96, 'max': 1400},
          'other': {'min': 1, 'mean': 81, 'max': 800},
          'nonprofit': {'min': 1, 'mean': 72, 'max': 1600},
          'education': {'min': 2, 'mean': 163, 'max': 1500},
          'technology': {'min': 5, 'mean': 455, 'max': 10000},
          'manufacturing': {'min': 2, 'mean': 200, 'max': 20000},
          'financial_services': {'min': 1, 'mean': 106, 'max': 3400},
          'retail': {'min': 2, 'mean': 240, 'max': 7500},
          'healthcare': {'min': 1, 'mean': 182, 'max': 9000},
          'professional_services': {'min': 1, 'mean': 90, 'max': 3600}
      }

      # Average crisis services cost per sector
      self.crisis_cost_per_sector = {
          'gaming_casino': 342,
          'telecommunications': 533,
          'entertainment': 124,
          'media': 77,
          'restaurant': 66,
          'energy': 73,
          'transportation': 86.7,
          'hospitality': 155,
          'public_entity': 72,
          'other': 62,
          'nonprofit': 71,
          'education': 114,
          'technology': 173,
          'manufacturing': 37,
          'financial_services': 78,
          'retail': 228,
          'healthcare': 157,
          'professional_services': 57
      }

    else:
      # Probability of a company being in a specific sector
      self.probability_of_belonging_to_a_sector = {
          'manufacturing': 0.03,
          'technology': 0.03,
          'energy': 0.04,
          'hospitality': 0.04,
          'professional_services': 0.04,
          'other': 0.09,
          'education': 0.08,
          'financial_services': 0.15,
          'retail': 0.24,
          'healthcare': 0.26
      }

      # Probability of a certain cause of loss for a company
      self.probability_of_a_cause_knowing_loss = {
          # social engineering also includes BEC, phishing, wire transfer fraud
          'company_trains_employees_against_social_engineering': 0.88,
          'company_has_patching_process': 0.74,
          'company_has_antivirus': 0.77
      }

      # Average cost in k$ per type of data exposed for a company
      self.cost_for_data_leaked = {
          'pci': 4900,
          'phi': 3200,
          'pii': 6500,
          'credentials': 575,
          'other': 1514
      }

      # Average crisis services cost per sector
      self.crisis_cost_per_sector = {
        'gaming_casino': 60,
        'telecommunications': 218,
        'entertainment': 258,
        'media': 258,
        'restaurant': 258,
        'energy': 258,
        'transportation': 258,
        'hospitality': 4100,
        'public_entity': 258,
        'other': 258,
        'nonprofit': 11,
        'education': 211,
        'technology': 1200,
        'manufacturing': 33000,
        'financial_services': 12500,
        'retail': 1800,
        'healthcare': 2400,
        'professional_services': 3100
      }

      # min, mean, max breach cost per sector
      self.breach_cost_per_sector = {
          'gaming_casino': {'min': 80, 'mean': 80, 'max': 80},
          'telecommunications': {'min': 400, 'mean': 400, 'max': 400},
          'energy': {'min': 2500, 'mean': 4200, 'max': 5000},
          'transportation': {'min': 80000, 'mean': 80000, 'max': 80000},
          'hospitality': {'min': 738, 'mean': 4200, 'max': 10000},
          'public_entity': {'min': 505, 'mean': 505, 'max': 505},
          'other': {'min': 100, 'mean': 219, 'max': 322},
          'nonprofit': {'min': 13, 'mean': 13, 'max': 13},
          'education': {'min': 3, 'mean': 216, 'max': 875},
          'technology': {'min': 1000, 'mean': 2600, 'max': 4100},
          'manufacturing': {'min': 20, 'mean': 16500, 'max': 33000},
          'financial_services': {'min': 72, 'mean': 10700, 'max': 64000},
          'retail': {'min': 60, 'mean': 4200, 'max': 16800},
          'healthcare': {'min': 5, 'mean': 3400, 'max': 15000},
          'professional_services': {'min': 332, 'mean': 3100, 'max': 6200}
      }

    self.probabilities_for_sectors_to_have_data = {
        'gaming_casino': {
            # Values are the probabilities that the attribute is True
            'pii': 0.84,
            'phi': 0.31,
            'pci': 0.25
        },
        'telecommunications': {
            # Values are the probabilities that the attribute is True
            'pii': 0.69,
            'credentials': 0.41,
            'other': 0.34
        },
        'entertainment': {
            # Values are the probabilities that the attribute is True
            'pii': 0.84,
            'phi': 0.31,
            'pci': 0.25
        },
        'media': {
            # Values are the probabilities that the attribute is True
            'pii': 0.69,
            'credentials': 0.41,
            'other': 0.34
        },
        'restaurant': {
            # Values are the probabilities that the attribute is True
            'pii': 0.44,
            'credentials': 0.14,
            'other': 0.10,
            'pci': 0.68
        },
        'energy': {
            # Values are the probabilities that the attribute is True
            'pii': 0.41,
            'credentials': 0.41,
            'other': 0.35,
            'pci': 0.68
        },
        'transportation': {
            # Values are the probabilities that the attribute is True
            'pii': 0.64,
            'credentials': 0.34,
            'other': 0.23
        },
        'hospitality': {
            # Values are the probabilities that the attribute is True
            'pii': 0.44,
            'credentials': 0.14,
            'other': 0.10,
            'pci': 0.68
        },
        'public_entity': {
            # Values are the probabilities that the attribute is True
            'pii': 0.51,
            'credentials': 0.33,
            'other': 0.34
        },
        'other': {
            # Values are the probabilities that the attribute is True
            'pii': 0.81,
            'credentials': 0.36,
            'other': 0.42
        },
        'nonprofit': {
            # Values are the probabilities that the attribute is True
            'pii': 0.81,
            'credentials': 0.36,
            'other': 0.42
        },
        'education': {
            # Values are the probabilities that the attribute is True
            'pii': 0.75,
            'credentials': 0.30,
            'other': 0.23
        },
        'technology': {
            # Values are the probabilities that the attribute is True
            'pii': 0.69,
            'credentials': 0.41,
            'other': 0.34
        },
        'manufacturing': {
            # Values are the probabilities that the attribute is True
            'pii': 0.49,
            'credentials': 0.55,
            'other': 0.25,
            'pci': 0.20
        },
        'financial_services': {
            # Values are the probabilities that the attribute is True
            'pii': 0.77,
            'credentials': 0.35,
            'other': 0.35,
            'pci': 0.32
        },
        'retail': {
            # Values are the probabilities that the attribute is True
            'pii': 0.49,
            'credentials': 0.27,
            'other': 0.25,
            'pci': 0.47
        },
        'healthcare': {
            # Values are the probabilities that the attribute is True
            'pii': 0.77,
            'phi': 67,
            'credentials': 0.18,
            'other': 0.18
        },
        'professional_services': {
            # Values are the probabilities that the attribute is True
            'pii': 0.75,
            'credentials': 0.45,
            'other': 0.32
        }
    }

    self.company_sector = self.DrawAttributeFromProbabilities(
        self.probability_of_belonging_to_a_sector)

    # Fill PII info for the features, since it depends on the company sector
    for type_of_data in ['pii', 'pci', 'phi', 'credentials', 'other']:
      if type_of_data in self.probabilities_for_sectors_to_have_data[
          self.company_sector]:
        self.features['company_has_{0:s}'.format(
            type_of_data)] = self.probabilities_for_sectors_to_have_data[
                self.company_sector][type_of_data]

    self.company_features = self.DrawFromProbabilities(self.features)

    self.loss_probability = self.ComputeLossProbability()

    self.potential_cost = self.ComputePotentialCost()

    self.company_features['company_type'] = self.company_type
    self.company_features['company_sector'] = self.company_sector

  @staticmethod
  def DrawAttributeFromProbabilities(probabilities: Dict[str, float]) -> str:
    """Draw an attribute based on associated probabilities.

    Args:
      probabilities (Dict[str, float]): The attributes and their probabilities.

    Returns:
      str: The chosen attribute.
    """
    sorted_probabilities = {k: v for k, v in sorted(  # pylint: disable=unnecessary-comprehension
        probabilities.items(), key=lambda item: item[1])}
    random_prob = np.random.uniform()
    previous_prob = 0.
    for key, value in sorted_probabilities.items():
      value += previous_prob
      if value >= random_prob:
        return key
      previous_prob = value
    return ''

  @staticmethod
  def DrawFromProbabilities(probabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Draw probabilities.

    Args:
      probabilities (Dict[str, float]): The probabilities to draw from.

    Returns:
      Dict[str, Any]: A dictionary with features being True or False.
    """
    choice = {}
    for prob in probabilities:
      if not probabilities[prob]:
        choice[prob] = 0
        continue
      random_prob = np.random.uniform()
      if 1 - probabilities[prob] >= random_prob:
        choice[prob] = 0
      else:
        choice[prob] = 1
    return choice

  def ComputeLossProbability(self) -> float:
    """Compute a loss probability for a given company.

    Returns:
      float: The loss probability.
    """

    # Probability that any company suffers from a loss
    p_loss = 0.20

    p_antivirus_knowing_loss = self.GetProbabilityForFeature(
        self.probability_of_a_cause_knowing_loss,
        'company_has_antivirus')
    p_training_knowing_loss = self.GetProbabilityForFeature(
        self.probability_of_a_cause_knowing_loss,
        'company_trains_employees_against_social_engineering')
    p_patching_knowing_loss = self.GetProbabilityForFeature(
        self.probability_of_a_cause_knowing_loss,
        'company_has_patching_process')

    p_antivirus = self.GetProbabilityForFeature(
        self.features, 'company_has_antivirus')  # type: ignore
    p_training = self.GetProbabilityForFeature(
        self.features,  # type: ignore
        'company_trains_employees_against_social_engineering')
    p_patching = self.GetProbabilityForFeature(
        self.features, 'company_has_patching_process')  # type: ignore

    numerator = p_antivirus_knowing_loss
    numerator *= p_training_knowing_loss * p_patching_knowing_loss * p_loss
    denominator = p_antivirus * p_training * p_patching

    loss = np.around((numerator / denominator), decimals=4)  # type: float
    loss = np.around(loss, decimals=4)
    if loss > 1.:
      loss = 1.
    if loss < 0.:
      loss = 0.
    return loss

  def GetProbabilityForFeature(self,
                               probabilities: Dict[str, float],
                               feature: str) -> float:
    """Return probability of a feature depending on company characteristics.

    Args:
      probabilities (Dict): Dictionary of probabilities for various features.
      feature (str): Key for the probabilities dictionary.

    Returns:
      float: The probability associated to the feature, depending if the
          company has it or not.
    """
    prob = probabilities[feature]
    company_has_it = self.company_features[feature]
    if prob and not company_has_it:
      prob = 1. - prob
    return prob

  def ComputePotentialCost(self) -> float:
    """Computes the potential cost for a breach/incident for this company.

    This includes recovery costs (crisis service costs).

    Returns:
      float: The cost of the breach.
    """
    initial_cost = 0.  # type: float
    potentially_exposed_data = self.probabilities_for_sectors_to_have_data[
        self.company_sector]
    for exposed_data in potentially_exposed_data:
      if self.company_features['company_has_{0:s}'.format(exposed_data)]:
        initial_cost += self.cost_for_data_leaked[exposed_data]
    initial_cost += self.crisis_cost_per_sector[self.company_sector]
    min_value = self.breach_cost_per_sector[self.company_sector]['min']
    max_value = self.breach_cost_per_sector[self.company_sector]['max']
    mean_value = self.breach_cost_per_sector[self.company_sector]['mean']
    if min_value == max_value:
      min_value -= 1
    rand = np.random.triangular(
        min_value, mean_value, max_value)
    initial_cost = np.around(initial_cost + rand, decimals=2)
    return initial_cost


if __name__ == '__main__':
  data = DataGenerator().GenerateData(n_samples=50000)
  data.to_csv('./src/old/synthetic_data_A_old.csv', index=False)
