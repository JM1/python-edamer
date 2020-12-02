#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>

from edamer import detail, dt, fn
import logging
from mpi4py import MPI
import numpy as np  # noqa F401
import pytest

TestArguments = dict(
    dataset=[
        detail.TestDB.MatrixA,
        detail.TestDB.MatrixG,
        detail.TestDB.MatrixJ,
        detail.TestDB.MatrixK,
        detail.TestDB.MatrixL,
        detail.TestDB.MatrixM,
        detail.TestDB.MatrixN,
        detail.TestDB.MatrixO,
        detail.TestDB.MatrixP
    ],
    factory=[
        # lambdas should return 'None' if factory does not support given input
        ("NumPy", lambda dataset:
            (
                np.transpose,
                np.asarray(dataset, order='F')
            )),
        ("ElMatrix", lambda dataset:
            (
                fn.transpose,
                dt.ElMatrix.view_from_numpy(np.asarray(dataset, order='F'))
            )),
        ("ElDistMatrix", lambda dataset:
            (
                fn.transpose,
                dt.ElDistMatrix.make_view(
                    dt.ElGrid(MPI.COMM_WORLD),
                    dt.ElMatrix.view_from_numpy(np.asarray(dataset, order='F')),
                    dt.MatrixDistribution.make(
                        dt.ElDist.STAR,
                        dt.ElDist.STAR,
                        dt.ElDistWrap.ELEMENT)
                )
            ))
    ]
)


@pytest.fixture
def env():
    class Environment():
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        grid = dt.ElGrid(comm)
    return Environment()


@pytest.mark.parametrize("dataset", TestArguments["dataset"])
def test_fn_transpose_compare_results(env, dataset):
    logging.debug('dataset:    %r' % type(dataset))

    results = []
    factories = TestArguments["factory"]

    for factory in factories:
        factory_n = factory[0]
        factory_f = factory[1]
        logging.debug('factory:    %r' % factory_n)

        testcase = factory_f(dataset)
        if testcase is None:
            logging.info("unsupported configuration")
            continue

        factory_result = testcase[0](*testcase[1:])
        results.append([factory_n, factory_result])

    for i in range(0, len(results)-1):
        j = i+1

        factory_n_i, factory_result_i = results[i]
        factory_n_j, factory_result_j = results[j]
        logging.debug("comparing results for impl %s and %s" % (factory_n_i, factory_n_j))
        assert detail.test.matrix_matrix_allclose(factory_result_i,  factory_result_j)
        logging.info("comparing impl %s and %s done." % (factory_n_i, factory_n_j))
