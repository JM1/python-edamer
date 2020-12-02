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
        n = 1000  # vector length
    return Environment()


def test_ctor_view(env):
    for dtype in detail.scalars():
        for vector_t in [dt.ElColumnVector, dt.ElRowVector]:
            vec_np = np.asarray(np.arange(env.n), order='F', dtype=dtype)
            assert vec_np[3] == 3

            vec_el = vector_t.view_from_numpy(vec_np)
            assert isinstance(vec_el, vector_t)
            assert vec_el.length() == vec_np.size
            vec_np2 = vec_el.view_to_numpy()
            assert vec_np.size == vec_np2.size
            assert not vec_np2.flags.owndata
            assert np.array_equal(vec_np, vec_np2)

            vec_np2[3] = 1337
            assert vec_np[3] == 1337


def test_ctor_view_readonly(env):
    # TODO: Find out why dt.El*Vector.view_from_numpy() throws "RuntimeError: Could not activate keep_alive!" for
    #       readonly NumPy arrays with dtype in detail.scalars().
    for vector_t in [dt.ElColumnVector, dt.ElRowVector]:
        vec_np = np.asarray(np.arange(env.n), order='F')
        vec_np.setflags(write=0)
        logging.debug(vec_np.flags)
        logging.debug(vec_np.dtype)
        logging.debug(str(vec_np))

        vec_el = vector_t.view_from_numpy(vec_np)
        assert vec_el.length() == vec_np.size
        vec_np2 = vec_el.view_to_numpy()
        logging.debug(vec_np2.flags)
        assert not vec_np2.flags.owndata
        assert np.array_equal(vec_np, vec_np2)


def test_ctor_complex(env):
    for dtype in detail.complex_scalars():
        for vector_t in [dt.ElColumnVector, dt.ElRowVector]:
            vec_np = np.asarray(np.arange(env.n), order='F', dtype=dtype)
            vec_el = vector_t.view_from_numpy(vec_np)
            vec_np2 = vec_el.view_to_numpy()

            vec_np2[3] = 1337 + 42j
            assert vec_np[3] == 1337 + 42j

            logging.debug(str(vec_np2))
            logging.debug(vec_np2.flags)
            logging.debug(vec_np2.dtype)
