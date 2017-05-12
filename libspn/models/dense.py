# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.models.model import Model
from libspn.log import get_logger
from libspn.graph.ivs import IVs


class DenseModel(Model):

    logger = get_logger()
    info = logger.info

    def __init__(self):
        super().__init__()

    def build_spn(self, dataset):
        self.info("Building SPN")
        # Get data from dataset
        data = dataset.get_data()
        print(data.shape)

        # IVs
        self._ivs = IVs(num_vars=self._num_vars, num_vals=3)

    def train(self):
        pass

    def test(self):
        pass
