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
        n = 2000  # vector length
    return Environment()


def test_ctor(env):
    if np.double not in detail.scalars():
        pytest.skip("unsupported configuration")

    for dist_vector_t in [
        dt.ElDistColumnVector_Double_ElSTAR_ElSTAR_ElELEMENT,
        dt.ElDistRowVector_Double_ElSTAR_ElSTAR_ElELEMENT,
        dt.ElDistColumnVector_Double_ElMC_ElMR_ElELEMENT,
        dt.ElDistRowVector_Double_ElMC_ElMR_ElELEMENT
    ]:
        v = dist_vector_t(env.grid, env.n)
    assert v.length() == env.n


def test_make(env):
    for dtype in detail.scalars() + detail.complex_scalars():
        for vector_t, dist_vector_t in [
            (dt.ElColumnVector, dt.ElDistColumnVector),
            (dt.ElRowVector, dt.ElDistRowVector)
        ]:
            vec_np = np.asarray(np.arange(env.n), order='F', dtype=dtype)
            vec_el = vector_t.view_from_numpy(vec_np)

            dvec_el = dist_vector_t.make_view(env.grid, vec_el)
            assert isinstance(dvec_el, dist_vector_t)

            assert dvec_el.length() == env.n


def test_local(env):
    for dtype in detail.scalars():  # TODO: Add detail.complex_scalars()
        for vector_t, dist_vector_t in [
            (dt.ElColumnVector, dt.ElDistColumnVector),
            (dt.ElRowVector, dt.ElDistRowVector)
        ]:
            vec_np = np.asarray(np.arange(env.n), order='F', dtype=dtype)
            assert vec_np[3] == 3
            vec_el = vector_t.view_from_numpy(vec_np)

            dvec_el = dist_vector_t.make_view(env.grid, vec_el)
            lcl_el = dvec_el.local()
            lcl_np = lcl_el.view_to_numpy()

            lcl_np[3] = -1337
            assert vec_np[3] == -1337


def test_copy(env):
    for dtype in detail.scalars():  # TODO: Add detail.complex_scalars()
        for vector_t, dist_vector_t in [
            (dt.ElColumnVector, dt.ElDistColumnVector),
            (dt.ElRowVector, dt.ElDistRowVector)
        ]:
            vec_np = np.asarray(np.arange(env.n), order='F', dtype=dtype)
            assert vec_np[3] == 3
            vec_el = vector_t.view_from_numpy(vec_np)

            dvec_el = dist_vector_t.make_view(env.grid, vec_el)

            dvec2_el = dvec_el.copy()

            lcl_el = dvec_el.local()
            lcl2_el = dvec2_el.local()

            lcl_np = lcl_el.view_to_numpy()
            lcl2_np = lcl2_el.view_to_numpy()
            assert lcl_np[3] == 3
            assert lcl2_np[3] == 3
            lcl_np[3] = -1337
            assert vec_np[3] == -1337
            assert lcl2_np[3] == 3


def test_copy_redist(env):
    for dtype in detail.scalars() + detail.complex_scalars():
        for vector_t, dist_vector_t in [
            (dt.ElColumnVector, dt.ElDistColumnVector),
            (dt.ElRowVector, dt.ElDistRowVector)
        ]:
            vec_np = np.asarray(np.arange(env.n), order='F', dtype=dtype)
            assert vec_np[3] == 3
            vec_el = vector_t.view_from_numpy(vec_np)

            dist_star_star_el = dt.MatrixDistribution.make(dt.ElDist.STAR, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)
            dist_mc_mr_el = dt.MatrixDistribution.make(dt.ElDist.MC, dt.ElDist.MR, dt.ElDistWrap.ELEMENT)
            dist_circ_circ_el = dt.MatrixDistribution.make(dt.ElDist.CIRC, dt.ElDist.CIRC, dt.ElDistWrap.ELEMENT)
            dist_vc_star_el = dt.MatrixDistribution.make(dt.ElDist.VC, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)

            dvec_star_star_el = dist_vector_t.make_view(env.grid, vec_el, dist_star_star_el)
            dvec_mc_mr_el = dvec_star_star_el.copy(dist_mc_mr_el)
            dvec_vc_star_el = dvec_mc_mr_el.copy(dist_vc_star_el)
            dvec_circ_circ_el = dvec_vc_star_el.copy(dist_circ_circ_el)
            dvec_star_star_el2 = dvec_circ_circ_el.copy(dist_star_star_el)

            lcl_star_star_el = dvec_star_star_el.local()
            lcl_star_star_el2 = dvec_star_star_el2.local()

            assert dvec_star_star_el.length() == env.n
            assert dvec_mc_mr_el.length() == env.n
            assert dvec_vc_star_el.length() == env.n
            assert dvec_circ_circ_el.length() == env.n
            assert dvec_star_star_el2.length() == env.n

            lcl_star_star_np = lcl_star_star_el.view_to_numpy()
            lcl_star_star_np2 = lcl_star_star_el2.view_to_numpy()

            assert np.array_equal(lcl_star_star_np, lcl_star_star_np2)
