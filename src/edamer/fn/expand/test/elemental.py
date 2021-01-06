#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>

from edamer import detail, dt, fn
from mpi4py import MPI
import numpy as np
import pytest


@pytest.fixture
def env():
    class Environment():
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        grid = dt.ElGrid(comm)
        m = 1000
        n = 2000
    return Environment()


def test_fn_expand(env):
    dist_star_star_el = dt.MatrixDistribution.make(dt.ElDist.STAR, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)

    mat_np = np.asarray(np.arange(env.m*env.n).reshape(env.m, env.n), dtype=np.float64, order='F')
    mat_el = dt.ElMatrix.view_from_numpy(mat_np)
    dmat_el = dt.ElDistMatrix.make_view(env.grid, mat_el, dist_star_star_el)

    vec_np = np.asarray(np.arange(env.n), dtype=np.float64, order='F')
    vec_el = dt.ElRowVector.view_from_numpy(vec_np)
    dvec_el = dt.ElDistRowVector.make_view(env.grid, vec_el, dist_star_star_el)

    assert fn.size(dmat_el).m == env.m
    assert fn.size(dmat_el).n == env.n
    assert fn.size(dvec_el) == env.n

    expr = fn.expand(dvec_el, fn.size(dmat_el))

    assert isinstance(expr, dt.Expression), "no instance of dt.Expression"
    assert expr.operation == fn.expand, "operation is not fn.expand"

    sum_el = fn.plus(dmat_el, expr)
    sum_np = np.add(mat_np, vec_np)
    assert detail.test.matrix_matrix_equal(sum_np, sum_el)
