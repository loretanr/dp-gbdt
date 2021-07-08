# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Class to generate synthetic data."""

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
    # print('Identifying distribution of stereotype company profiles...')
    # stereotypes_distribution = self.GetStereotypeDistribution(
    #     n_samples=n_samples)

    print('Generating {0:d} samples...'.format(n_samples))
    samples = []  # type: List[List[Union[int, float]]]

    while len(samples) < n_samples:
      stereotype = CompanyProfileStereotype()
      # Each stereotype will make up for 1% of the final data
      n_samples_stereotype = int(np.ceil(n_samples / 100))
      samples.extend(self.GenerateSamplesFromStereotype(
          stereotype, n_samples_stereotype))

    samples = samples[:n_samples]

    header = list(CompanyProfileStereotype().company_profile.keys())

    # Shuffle samples, and return a dataframe with the generated data
    print('Shuffling generated data...')
    return pd.DataFrame(data=samples, columns=header).sample(
        frac=1).reset_index(drop=True)

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
        mean_x=stereotype.company_profile['company_suffers_a_loss'],
        std_x=((stereotype.company_profile['company_suffers_a_loss']/100)*15))
    cost_log_normal_mean, cost_log_normal_std = self.GetLogNormalParams(
        mean_x=stereotype.company_profile['cost_of_loss'],
        std_x=((stereotype.company_profile['cost_of_loss']/100)*15))

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

      sample.extend(list(stereotype.company_profile.values())[:-2])
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
    log_normal_std = np.sqrt(np.log(1 + (
        np.square(std_x) / np.square(mean_x))))
    return log_normal_mean, log_normal_std


class CompanyProfileStereotype:
  """Class that represents a company profile stereotype."""
  # pylint: disable=invalid-name

  def __init__(self) -> None:

    #  company_has_pii, company_has_pci, company_has_phi,
    #  company_has_credentials, company_has_other are filled at runtime since
    #  it is size and sector dependant (not all profiles have all kind of data)
    self.company_profile = {

        'company_size': None,
        'company_sector': None,
        'company_allows_BYOD': None,
        'company_does_yearly_pentest': None,
        'company_uses_RBAC': None,
        'company_does_red_teaming': None,
        'company_has_2FA': None,
        'company_has_password_policy': None,
        'company_trains_employees_against_social_engineering': None,
        'company_has_antivirus': None,
        'company_has_patching_process': None,
        'company_has_ingress_firewall': None,
        'company_has_egress_firewall': None,
        'company_uses_3rd_party': None,
        'company_has_ciso': None,
        'company_has_threat_intelligence_team': None,
        'company_monitors_IOCs': None,
        'company_has_IDS': None,
        'company_has_IPS': None,
        'company_separates_systems': None,
        'company_does_daily_backups': None,
        'company_has_recovery_plan': None,
        'company_has_incident_response_team': None
    }  # type: Dict[str, Any]

    # Probabilities of company features
    #  company_has_pii, company_has_pci, company_has_phi,
    #  company_has_credentials, company_has_other are filled at runtime since
    #  it is size and sector dependant (not all profiles have all kind of data)
    self.features_probabilities = {
        'company_allows_BYOD': 0.44,
        'company_does_yearly_pentest': 0.36,
        'company_uses_RBAC': 0.85,
        'company_does_red_teaming': 0.33,
        'company_has_2FA': 0.78,
        'company_has_password_policy': 0.88,
        'company_trains_employees_against_social_engineering': 0.70,
        'company_has_antivirus': 0.95,
        'company_has_patching_process': 0.60,
        'company_has_ingress_firewall': 0.83,
        'company_has_egress_firewall': 0.61,
        'company_uses_3rd_party': 0.87,
        'company_has_ciso': 0.67,
        'company_has_IDS': 0.94,
        'company_has_IPS': 0.81,
        'company_has_threat_intelligence_team': 0.38,
        'company_has_incident_response_team': 0.46,
        'company_monitors_IOCs': 0.44,
        'company_separates_systems': 0.79,
        'company_does_daily_backups': 0.91,
        'company_has_recovery_plan': 0.33,
    }  # type: Dict[str, float]

    # Probability of being a large or small company
    # A small company is a company with less than 2 billions revenue
    self.company_size_probabilities = {
        'small': 0.68,
        'large': 0.32
    }

    # Assign company size
    self.company_profile['company_size'] = self.DrawAttributeFromProbabilities(
        self.company_size_probabilities)

    if self.company_profile['company_size'] == 'small':
      # Probability of a company being in a specific sector
      # https://netdiligence.com/wp-content/uploads/2020/05/2019_NetD_Claims_
      # Study_Report_1.2.pdf
      self.company_sector_knowing_size_probabilities = {
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
      self.company_sector_knowing_size_probabilities = {
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

    # Assign company sector
    self.company_profile[
        'company_sector'] = self.DrawAttributeFromProbabilities(
            self.company_sector_knowing_size_probabilities)

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
            'phi': 0.67,
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

    # Assign types of data that the company might store
    for type_of_data in ['pii', 'pci', 'phi', 'credentials', 'other']:
      if type_of_data in self.probabilities_for_sectors_to_have_data[
          self.company_profile['company_sector']]:
        self.features_probabilities['company_has_{0:s}'.format(
            type_of_data)] = self.probabilities_for_sectors_to_have_data[
                self.company_profile['company_sector']][type_of_data]
      else:
        self.features_probabilities['company_has_{0:s}'.format(
            type_of_data)] = 0.

    # Given a loss, how likely is the feature to be True.
    # This data is extracted from the reports, or made up if information
    # wasn't available.
    self.single_feature_knowing_loss_probabilities = {
        'company_size_and_sector':
            self.company_sector_knowing_size_probabilities[
                self.company_profile['company_sector']],
        'company_allows_BYOD': 0.14,
        'company_does_yearly_pentest': 0.23,
        'company_uses_RBAC': 0.67,
        'company_has_patching_process': 0.828,
        'company_has_antivirus': 0.81,
        'company_has_ingress_firewall': 0.84,
        'company_has_egress_firewall': 0.76,
        'company_uses_3rd_party': 0.79,
        'company_has_ciso': 0.68,
        # 'company_does_red_teaming': 0.12,
        # 'company_has_2FA': 0.64,
        # 'company_has_password_policy': 0.71,
        # 'company_trains_employees_against_social_engineering': 0.8025,
        # 'company_has_threat_intelligence_team': 0.24,
        # 'company_monitors_IOCs': 0.39,
        # 'company_has_IDS': 0.89,
        # 'company_has_IPS': 0.75,
    }  # type: Dict[str, float]

    # Given a loss, how likely are the features to be True/False.
    # This data is extracted from the reports, or made up if information
    # wasn't available.
    # Each sub-dictionary's key is the attribute being True or False,
    # in order.
    self.multiple_features_knowing_loss_probabilities = {
        # Red teaming, 2FA, Password Policy, Social engineering training
        'R2PS': {
            # Red teaming True
            1: {
                # 2FA True
                1: {
                    # Password policy True
                    1: {
                        # Social engineering training True
                        1: 0.08,
                        # Social engineering Training False
                        0: 0.13
                    },
                    # Password policy False
                    0: {
                        # Social engineering training True
                        1: 0.16,
                        # Social engineering Training False
                        0: 0.24
                    }
                },
                # 2FA False
                0: {
                    # Password policy True
                    1: {
                        # Social engineering training True
                        1: 0.04,
                        # Social engineering Training False
                        0: 0.31
                    },
                    # Password policy False
                    0: {
                        # Social engineering training True
                        1: 0.17,
                        # Social engineering Training False
                        0: 0.03
                    }
                }
            },
            # Red teaming False
            0: {
                # 2FA True
                1: {
                    # Password policy True
                    1: {
                        # Social engineering training True
                        1: 0.52,
                        # Social engineering Training False
                        0: 0.62
                    },
                    # Password policy False
                    0: {
                        # Social engineering training True
                        1: 0.48,
                        # Social engineering Training False
                        0: 0.37
                    }
                },
                # 2FA False
                0: {
                    # Password policy True
                    1: {
                        # Social engineering training True
                        1: 0.71,
                        # Social engineering Training False
                        0: 0.84
                    },
                    # Password policy False
                    0: {
                        # Social engineering training True
                        1: 0.59,
                        # Social engineering Training False
                        0: 0.92
                    }
                }
            }
        },
        # IPS, IDS, Threat Intelligence, IOCs
        'PDTI': {
            # IPS True
            1: {
                # IDS True
                1: {
                    # Threat Intelligence True
                    1: {
                        # IOCs True
                        1: 0.07,
                        # IOCs False
                        0: 0.18
                    },
                    # Threat Intelligence False
                    0: {
                        # IOCs True
                        1: 0.09,
                        # IOCs False
                        0: 0.15
                  }
                },
                # IDS False
                0: {
                    # Threat Intelligence True
                    1: {
                        # IOCs True
                        1: 0.11,
                        # IOCs False
                        0: 0.22
                    },
                    # Threat Intelligence False
                    0: {
                        # IOCs True
                        1: 0.29,
                        # IOCs False
                        0: 0.47
                    }
                }
            },
            # IPS False
            0: {
                # IDS True
                1: {
                    # Threat Intelligence True
                    1: {
                      # IOCs True
                      1: 0.09,
                      # IOCs False
                      0: 0.14
                    },
                    # Threat Intelligence False
                    0: {
                      # IOCs True
                      1: 0.27,
                      # IOCs False
                      0: 0.15
                    }
                },
                # IDS False
                0: {
                    # Threat Intelligence True
                    1: {
                        # IOCs True
                        1: 0.12,
                        # IOCs False
                        0: 0.08
                    },
                    # Threat Intelligence False
                    0: {
                        # IOCs True
                        1: 0.18,
                        # IOCs False
                        0: 0.84
                    }
                }
            }
        }
    }

    # Given some features, how likely is another feature to be True.
    # Each sub-dictionary's key is the attribute being True or False,
    # in order.
    # In P(A|B), the first key in this dictionary is A.
    self.single_feature_knowing_other_probabilities = {
        'PWD': {
            1: {
                '2FA': {
                    1: 0.91,
                    0: 0.84
                }
            },
            0: {
                '2FA': {
                    1: 0.15,
                    0: 0.61
                }
            }
        },
        'SOC': {
            1: {
                'PWD': {
                    1: {
                        'RED': {
                            1: 0.97,
                            0: 0.81
                        }
                    },
                    0: {
                        'RED': {
                            1: 0.85,
                            0: 0.72
                        }
                    }
                }
            },
            0: {
                'PWD': {
                    1: {
                        'RED': {
                            1: 0.06,
                            0: 0.58
                        }
                    },
                    0: {
                        'RED': {
                            1: 0.02,
                            0: 0.44
                        }
                    }
                }
            }
        },
        'IPS': {
            1: {
                'IDS': {
                    1: 0.97,
                    0: 0.75
                }
            },
            0: {
                'IDS': {
                    1: 0.28,
                    0: 0.54
                }
            }
        },
        'THREAT': {
            1: {
                'IDS': {
                    1: 0.79,
                    0: 0.31
                }
            },
            0: {
                'IDS': {
                    1: 0.48,
                    0: 0.88
                }
            }
        },
        'IOC': {
            1: {
                'THREAT': {
                    1: 0.93,
                    0: 0.74
                }
            },
            0: {
                'THREAT': {
                    1: 0.22,
                    0: 0.39
                }
            }
        },
        'IR': {
            1: {
                'THREAT': {
                    1: {
                        'IDS': {
                            1: {
                                'IPS': {
                                    1: 0.88,
                                    0: 0.76
                                }
                            },
                            0: {
                                'IPS': {
                                    1: 0.78,
                                    0: 0.62
                                }
                            }
                        }
                    },
                    0: {
                        'IDS': {
                            1: {
                                'IPS': {
                                  1: 0.69,
                                  0: 0.53
                                }
                            },
                            0: {
                                'IPS': {
                                  1: 0.57,
                                  0: 0.41
                                }
                            }
                        }
                    }
                }
            }
        }
    }  # type: Dict[str, Any]

    self.AssignFeaturesToCompanyProfile(self.company_profile,
                                        self.features_probabilities)

    self.company_profile[
        'company_suffers_a_loss'] = self.ComputeLossProbability()

    self.company_profile['cost_of_loss'] = self.ComputePotentialCost()

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

  def AssignFeaturesToCompanyProfile(
      self,
      company_profile: Dict[str, Any],
      probabilities: Dict[str, float]) -> None:
    """Draw probabilities.

    Args:
      company_profile (Dict[str, Any]): The company profile to fill.
      probabilities (Dict[str, float]): The probabilities to draw from.
    """

    # Let's define some of the conditional probabilities

    def get_pwd_policy() -> float:
      """Return probability for password policy."""
      has_2FA = company_profile['company_has_2FA']
      p = self.single_feature_knowing_other_probabilities['PWD'][1][
          '2FA'][has_2FA]  # type: float
      return p

    def get_soc() -> float:
      """Return probability for social engineering training."""
      has_pwd_policy = company_profile['company_has_password_policy']
      does_red_teaming = company_profile['company_does_red_teaming']
      p = self.single_feature_knowing_other_probabilities['SOC'][1][
          'PWD'][has_pwd_policy]['RED'][does_red_teaming]  # type: float
      return p

    def get_ips() -> float:
      """Return probability for IPS."""
      has_ids = company_profile['company_has_IDS']
      p = self.single_feature_knowing_other_probabilities['IPS'][1][
          'IDS'][has_ids]  # type: float
      return p

    def get_threat() -> float:
      """Return probability for threat intelligence."""
      has_ids = company_profile['company_has_IDS']
      p = self.single_feature_knowing_other_probabilities['THREAT'][1][
          'IDS'][has_ids]  # type: float
      return p

    def get_ioc() -> float:
      """Return probability for IOCs."""
      has_threat = company_profile['company_has_threat_intelligence_team']
      p = self.single_feature_knowing_other_probabilities['IOC'][1][
          'THREAT'][has_threat]  # type: float
      return p

    def get_ir() -> float:
      has_ids = company_profile['company_has_IDS']
      has_ips = company_profile['company_has_IPS']
      has_threat = company_profile['company_has_threat_intelligence_team']
      p = self.single_feature_knowing_other_probabilities['IR'][1]['THREAT'][
          has_threat]['IDS'][has_ids]['IPS'][has_ips]  # type: float
      return p

    for feature, probability in probabilities.items():
      if feature == 'company_has_password_policy':
        probability = get_pwd_policy()
      if feature == 'company_trains_employees_against_social_engineering':
        probability = get_soc()
      if feature == 'company_has_IPS':
        probability = get_ips()
      if feature == 'company_has_threat_intelligence_team':
        probability = get_threat()
      if feature == 'company_monitors_IOCs':
        probability = get_ioc()
      if feature == 'company_has_incident_response_team':
        probability = get_ir()
      random_prob = np.random.uniform()
      if probability >= random_prob:
        company_profile[feature] = 1
      else:
        company_profile[feature] = 0

  def ComputeLossProbability(self) -> float:
    """Compute a loss probability for a given company.

    Returns:
      float: The loss probability.
    """

    # Probability that any company suffers from a loss
    p_loss = 0.10

    does_red_teaming = self.company_profile['company_does_red_teaming']
    has_2FA = self.company_profile['company_has_2FA']
    has_pwd_policy = self.company_profile['company_has_password_policy']
    has_soc_training = self.company_profile[
        'company_trains_employees_against_social_engineering']
    has_IPS = self.company_profile['company_has_IPS']
    has_IDS = self.company_profile['company_has_IDS']
    has_threat = self.company_profile['company_has_threat_intelligence_team']
    has_IOC = self.company_profile['company_monitors_IOCs']

    allows_BYOD = self.company_profile['company_allows_BYOD']
    does_pentest = self.company_profile['company_does_yearly_pentest']
    uses_RBAC = self.company_profile['company_uses_RBAC']
    has_patching_process = self.company_profile['company_has_patching_process']
    has_antivirus = self.company_profile['company_has_antivirus']
    has_fw_ing = self.company_profile['company_has_ingress_firewall']
    has_fw_eg = self.company_profile['company_has_egress_firewall']
    uses_3rd_party = self.company_profile['company_uses_3rd_party']
    has_ciso = self.company_profile['company_has_ciso']

    numerator = self.GetProbability('company_size_and_sector',
                                    attribute_mask=1,
                                    knowing_loss=True)
    numerator *= self.GetProbability('company_allows_BYOD',
                                     attribute_mask=allows_BYOD,
                                     knowing_loss=True)
    numerator *= self.GetProbability('company_does_yearly_pentest',
                                     attribute_mask=does_pentest,
                                     knowing_loss=True)
    numerator *= self.GetProbability('company_uses_RBAC',
                                     attribute_mask=uses_RBAC,
                                     knowing_loss=True)

    numerator *= self.multiple_features_knowing_loss_probabilities['R2PS'][
        does_red_teaming][has_2FA][has_pwd_policy][has_soc_training]

    numerator *= self.GetProbability('company_has_patching_process',
                                     attribute_mask=has_patching_process,
                                     knowing_loss=True)
    numerator *= self.GetProbability('company_has_antivirus',
                                     attribute_mask=has_antivirus,
                                     knowing_loss=True)
    numerator *= self.GetProbability('company_has_ingress_firewall',
                                     attribute_mask=has_fw_ing,
                                     knowing_loss=True)
    numerator *= self.GetProbability('company_has_egress_firewall',
                                     attribute_mask=has_fw_eg,
                                     knowing_loss=True)
    numerator *= self.GetProbability('company_uses_3rd_party',
                                     attribute_mask=uses_3rd_party,
                                     knowing_loss=True)
    numerator *= self.GetProbability('company_has_ciso',
                                     attribute_mask=has_ciso,
                                     knowing_loss=True)

    numerator *= self.multiple_features_knowing_loss_probabilities['PDTI'][
        has_IPS][has_IDS][has_threat][has_IOC]

    numerator *= p_loss

    #################

    denominator = self.GetProbability('company_size_and_sector',
                                       attribute_mask=1,
                                       knowing_loss=True)
    denominator *= self.company_size_probabilities[
        self.company_profile['company_size']]
    denominator *= self.GetProbability('company_allows_BYOD',
                                       attribute_mask=allows_BYOD)
    denominator *= self.GetProbability('company_does_yearly_pentest',
                                       attribute_mask=does_pentest)
    denominator *= self.GetProbability('company_uses_RBAC',
                                       attribute_mask=uses_RBAC)
    denominator *= self.GetProbability('company_does_red_teaming',
                                       attribute_mask=does_red_teaming)
    denominator *= self.GetProbability('company_has_2FA',
                                       attribute_mask=has_2FA)

    denominator *= self.single_feature_knowing_other_probabilities['PWD'][
        has_pwd_policy]['2FA'][has_2FA]
    denominator *= self.single_feature_knowing_other_probabilities['SOC'][
        has_soc_training]['PWD'][has_pwd_policy]['RED'][does_red_teaming]

    denominator *= self.GetProbability('company_has_patching_process',
                                       attribute_mask=has_patching_process)
    denominator *= self.GetProbability('company_has_antivirus',
                                       attribute_mask=has_antivirus)
    denominator *= self.GetProbability('company_has_ingress_firewall',
                                       attribute_mask=has_fw_ing)
    denominator *= self.GetProbability('company_has_egress_firewall',
                                       attribute_mask=has_fw_eg)
    denominator *= self.GetProbability('company_uses_3rd_party',
                                       attribute_mask=uses_3rd_party)
    denominator *= self.GetProbability('company_has_ciso',
                                       attribute_mask=has_ciso)

    denominator *= self.single_feature_knowing_other_probabilities['IPS'][
        has_IPS]['IDS'][has_IDS]

    denominator *= self.GetProbability('company_has_IDS',
                                       attribute_mask=has_IDS)

    denominator *= self.single_feature_knowing_other_probabilities['THREAT'][
        has_threat]['IDS'][has_IDS]
    denominator *= self.single_feature_knowing_other_probabilities['IOC'][
        has_IOC]['THREAT'][has_threat]

    loss = np.divide(numerator, denominator)  # type: float
    if loss > 1.:
      loss = 1.

    return loss

  def ComputePotentialCost(self) -> float:
    """Computes the potential cost for a breach/incident for this company.

    This includes recovery costs (crisis service costs).

    Returns:
      float: The cost of the breach.
    """
    data_leaked_cost = 0.  # type: float
    potentially_exposed_data = self.probabilities_for_sectors_to_have_data[
        self.company_profile['company_sector']]
    for exposed_data in potentially_exposed_data:
      if self.company_profile['company_has_{0:s}'.format(exposed_data)]:
        data_leaked_cost += self.cost_for_data_leaked[exposed_data]

    has_system_sep = self.company_profile['company_separates_systems']
    does_daily_backup = self.company_profile['company_does_daily_backups']
    has_recovery_plan = self.company_profile['company_has_recovery_plan']
    has_ir_team = self.company_profile['company_has_incident_response_team']

    # Compute fraction of crisis and breach cost, depending on company profile
    fraction_crisis_cost = 1.
    fraction_breach_cost = 1.
    if has_ir_team:
      fraction_crisis_cost -= 0.24
    else:
      fraction_crisis_cost += 0.21
    if has_system_sep:
      fraction_crisis_cost -= 0.37
      fraction_breach_cost -= 0.53
    else:
      fraction_crisis_cost += 0.34
      fraction_breach_cost += 0.51
    if does_daily_backup:
      fraction_crisis_cost -= 0.36
      fraction_breach_cost -= 0.27
    else:
      fraction_crisis_cost += 0.34
      fraction_breach_cost += 0.20
    if has_recovery_plan:
      fraction_crisis_cost -= 0.57
    else:
      fraction_crisis_cost += 0.52

    if fraction_crisis_cost < 0.:
      fraction_crisis_cost = 0.
    if fraction_breach_cost < 0.:
      fraction_breach_cost = 0.

    crisis_cost = (self.crisis_cost_per_sector[
        self.company_profile['company_sector']] * fraction_crisis_cost)

    breach_cost = (self.breach_cost_per_sector[
        self.company_profile['company_sector']]['mean']) * fraction_breach_cost

    min_breach_cost = self.breach_cost_per_sector[
        self.company_profile['company_sector']]['min']
    max_breach_cost = self.breach_cost_per_sector[
        self.company_profile['company_sector']]['max']

    if breach_cost > max_breach_cost:
      breach_cost = max_breach_cost
    if breach_cost < min_breach_cost:
      breach_cost = min_breach_cost

    cost = np.around(
        data_leaked_cost + breach_cost + crisis_cost, decimals=2)  # type: float

    return cost

  def GetProbability(self, name: str,
                     attribute_mask: int,
                     knowing_loss: bool = False) -> float:
    """Get probability value.

    Args:
      name (str): The name of the probability to get.
      attribute_mask (int): 1 or 0, depending on company profile. If it's a
          1, then the company has that attribute.
      knowing_loss (bool): Optional. If True, returns p(name|loss=True).
          Default is False.

    Returns:
      float: A probability.
    """
    if knowing_loss:
      p = self.single_feature_knowing_loss_probabilities[name]
    else:
      p = self.features_probabilities[name]
    if attribute_mask:
      return p
    return 1. - p


if __name__ == '__main__':
  for l in ['A', 'B', 'C', 'D']:
    generator = DataGenerator()
    data = generator.GenerateData(n_samples=1000000)
    data.to_csv('./src/synthetic/synthetic_{0:s}.csv'.format(l),
                index=False)
