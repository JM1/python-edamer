#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>

from edamer import detail, dt
import logging # noqa F401
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
        m = 1000  # matrix height
        n = 2000  # matrix width
    return Environment()


def test_ctor(env):
    if np.double not in detail.scalars():
        pytest.skip("unsupported configuration")

    dt.ElDistMatrix_Double_ElSTAR_ElSTAR_ElELEMENT(env.grid, 2, 3).size().n


def test_make_view(env):
    for dtype in detail.scalars() + detail.complex_scalars():
        mat_np = np.asarray(np.arange(env.m*env.n).reshape(env.m, env.n), order='F', dtype=dtype)
        mat_el = dt.ElMatrix.view_from_numpy(mat_np)

        dist_el = dt.MatrixDistribution.make(
            dt.ElDist.STAR, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)
        dmat_el = dt.ElDistMatrix.make_view(env.grid, mat_el, dist_el)
        assert isinstance(dmat_el, dt.ElDistMatrix)

        assert dmat_el.size().m == env.m
        assert dmat_el.size().n == env.n


def test_local(env):
    for dtype in detail.scalars():  # TODO: Add detail.complex_scalars()
        mat_np = np.asarray(np.arange(env.m*env.n).reshape(env.m, env.n), order='F', dtype=dtype)
        assert mat_np[1, 3] == 2*env.m+3
        mat_el = dt.ElMatrix.view_from_numpy(mat_np)

        dist_el = dt.MatrixDistribution.make(
            dt.ElDist.STAR, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)
        dmat_el = dt.ElDistMatrix.make_view(env.grid, mat_el, dist_el)

        lcl_el = dmat_el.local()
        lcl_np = lcl_el.view_to_numpy()

        lcl_np[1, 3] = -1337
        assert mat_np[1, 3] == -1337


def test_copy(env):
    for dtype in detail.scalars():  # TODO: Add detail.complex_scalars()
        mat_np = np.asarray(np.arange(env.m*env.n).reshape(env.m, env.n), order='F', dtype=dtype)
        assert mat_np[1, 3] == 2*env.m+3
        mat_el = dt.ElMatrix.view_from_numpy(mat_np)

        dist_el = dt.MatrixDistribution.make(
            dt.ElDist.STAR, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)
        dmat_el = dt.ElDistMatrix.make_view(env.grid, mat_el, dist_el)

        dmat2_el = dmat_el.copy()

        lcl_el = dmat_el.local()
        lcl2_el = dmat2_el.local()

        lcl_np = lcl_el.view_to_numpy()
        lcl2_np = lcl2_el.view_to_numpy()
        assert lcl_np[1, 3] == 2*env.m+3
        assert lcl2_np[1, 3] == 2*env.m+3
        lcl_np[1, 3] = -1337
        assert mat_np[1, 3] == -1337
        assert lcl2_np[1, 3] == 2*env.m+3


def test_copy_redist(env):
    for dtype in detail.scalars() + detail.complex_scalars():
        mat_np = np.asarray(np.arange(env.m*env.n).reshape(env.m, env.n), order='F', dtype=dtype)
        assert mat_np[1, 3] == 2*env.m+3
        mat_el = dt.ElMatrix.view_from_numpy(mat_np)

        dist_star_star_el = dt.MatrixDistribution.make(dt.ElDist.STAR, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)
        dist_mc_mr_el = dt.MatrixDistribution.make(dt.ElDist.MC, dt.ElDist.MR, dt.ElDistWrap.ELEMENT)
        dist_circ_circ_el = dt.MatrixDistribution.make(dt.ElDist.CIRC, dt.ElDist.CIRC, dt.ElDistWrap.ELEMENT)
        dist_vc_star_el = dt.MatrixDistribution.make(dt.ElDist.VC, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)

        dmat_star_star_el = dt.ElDistMatrix.make_view(env.grid, mat_el, dist_star_star_el)
        dmat_mc_mr_el = dmat_star_star_el.copy(dist_mc_mr_el)
        dmat_vc_star_el = dmat_mc_mr_el.copy(dist_vc_star_el)
        dmat_circ_circ_el = dmat_vc_star_el.copy(dist_circ_circ_el)

        lcl_star_star_el = dmat_star_star_el.local()
        lcl_mc_mr_el = dmat_mc_mr_el.local()
        lcl_vc_star_el = dmat_vc_star_el.local()
        lcl_circ_circ_el = dmat_circ_circ_el.local()

        assert dmat_star_star_el.size().m == env.m
        assert dmat_star_star_el.size().n == env.n
        assert dmat_mc_mr_el.size().m == env.m
        assert dmat_mc_mr_el.size().n == env.n
        assert dmat_vc_star_el.size().m == env.m
        assert dmat_vc_star_el.size().n == env.n
        assert dmat_circ_circ_el.size().m == env.m
        assert dmat_circ_circ_el.size().n == env.n

        if env.size > 1:
            assert lcl_vc_star_el.size().m < lcl_star_star_el.size().m
            assert lcl_vc_star_el.size().n == lcl_star_star_el.size().n

        lcl_star_star_np = lcl_star_star_el.view_to_numpy()
        lcl_mc_mr_np = lcl_mc_mr_el.view_to_numpy() # noqa F841 # TODO: Add assert
        lcl_vc_star_np = lcl_vc_star_el.view_to_numpy() # noqa F841 # TODO: Add assert
        lcl_circ_circ_np = lcl_circ_circ_el.view_to_numpy()

        if env.rank == 0:
            # If this assertion fails due to zeros in the upper matrix indices, then you may try
            # to use a different MPI point-to-point management layer, e.g. ob1 instead of ucx.
            # To do so, set and export the OMPI_MCA_pml variable before executing the unit tests:
            #  export OMPI_MCA_pml=ob1
            assert np.array_equal(lcl_star_star_np, lcl_circ_circ_np)
