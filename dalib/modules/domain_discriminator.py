"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import List, Dict
import torch.nn as nn

__all__ = ['DomainDiscriminator']


class DomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: List, batch_norm=True):
        hidden_size.insert(0, in_feature)
        linear_layer_list = []
        layer_num = len(hidden_size)

        if batch_norm:
            for i in range(layer_num - 1):
                linear_layer_list.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
                linear_layer_list.append(nn.BatchNorm1d(hidden_size[i + 1]))
                linear_layer_list.append(nn.ReLU())
            linear_layer_list.append(nn.Linear(hidden_size[-1], 1))
            linear_layer_list.append(nn.Sigmoid())
            super(DomainDiscriminator, self).__init__(*linear_layer_list)
        else:
            for i in range(layer_num - 1):
                linear_layer_list.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
                linear_layer_list.append(nn.ReLU(inplace=True))
                linear_layer_list.append(nn.Dropout(0.5))
            linear_layer_list.append(nn.Linear(hidden_size[-1], 1))
            linear_layer_list.append(nn.Sigmoid())
            super(DomainDiscriminator, self).__init__(*linear_layer_list)

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]
