#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>

import edamer
import pytest


def test_make():
    with pytest.raises(edamer.dt.MatrixDistributionNotSupportedException):
        edamer.dt.MatrixDistribution.make(edamer.dt.ElDist.MC, edamer.dt.ElDist.MR, edamer.dt.ElDistWrap.BLOCK)

    mat_dist = edamer.dt.MatrixDistribution.make(edamer.dt.ElDist.MC, edamer.dt.ElDist.MR, edamer.dt.ElDistWrap.ELEMENT)
    assert isinstance(mat_dist, edamer.dt.MatrixDistribution)
