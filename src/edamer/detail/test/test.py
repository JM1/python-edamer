#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>

import edamer
import numpy as np


def test_arrays():
    np_a = np.array([[12.0, -51.0,   4.0],
                     [06.0, 167.0, -68.0],
                     [-4.0,  24.0, -41.0]])

    assert np.array_equal(np_a, edamer.detail.TestDB.MatrixA)
    assert edamer.detail.matrix_matrix_allclose(np_a, edamer.detail.TestDB.MatrixA)
