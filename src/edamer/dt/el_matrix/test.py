#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>

from edamer import detail, dt
import logging
import numpy as np
import pytest


@pytest.fixture
def env():
    class Environment():
        m = 1000  # matrix height
        n = 2000  # matrix width
    return Environment()


def test_ctor_view(env):
    for dtype in detail.scalars():
        mat_np = np.asarray(np.arange(env.m*env.n).reshape(env.m, env.n), order='F', dtype=dtype)
        assert mat_np[3, 5] == (4-1)*env.n+5
        logging.debug(mat_np.flags)
        logging.debug(mat_np.dtype)
        logging.debug(str(mat_np))
        mat_el = dt.ElMatrix.view_from_numpy(mat_np)
        assert isinstance(mat_el, dt.ElMatrix)
        assert mat_el.size().m, mat_el.size().n == mat_np.size
        mat_np2 = mat_el.view_to_numpy()
        assert not mat_np2.flags.owndata
        logging.debug(mat_np2.flags)
        logging.debug(str(mat_np2))
        assert np.array_equal(mat_np, mat_np2)

        mat_np2[3, 5] = 1337
        assert mat_np[3, 5] == 1337
        logging.debug(str(mat_np))


def test_ctor_view_readonly(env):
    # TODO: Find out why dt.ElMatrix.view_from_numpy() throws "RuntimeError: Could not activate keep_alive!" for
    #       readonly NumPy arrays with dtype in detail.scalars().
    mat_np = np.asarray(np.arange(env.m*env.n).reshape(env.m, env.n), order='F')
    mat_np.setflags(write=0)
    logging.debug(mat_np.flags)
    logging.debug(mat_np.dtype)
    logging.debug(str(mat_np))

    mat_el = dt.ElMatrix.view_from_numpy(mat_np)
    assert mat_el.size().m, mat_el.size().n == mat_np.size
    mat_np2 = mat_el.view_to_numpy()
    logging.debug(mat_np2.flags)
    assert not mat_np2.flags.owndata
    assert np.array_equal(mat_np, mat_np2)


def test_ctor_complex(env):
    for dtype in detail.complex_scalars():
        mat_np = np.asarray(np.arange(env.m*env.n).reshape(env.m, env.n), order='F', dtype=dtype)
        mat_el = dt.ElMatrix.view_from_numpy(mat_np)
        mat_np2 = mat_el.view_to_numpy()

        mat_np2[1, 3] = 1337 + 42j
        assert mat_np[1, 3] == 1337 + 42j

        logging.debug(str(mat_np2))
        logging.debug(mat_np2.flags)
        logging.debug(mat_np2.dtype)
