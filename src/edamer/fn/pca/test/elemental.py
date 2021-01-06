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
    economy=[True, False],
    center=[True, False],
    normalize=[True, False],
    factory=[
        # lambdas should return 'None' if factory does not support given input
        ("ElMatrix", lambda dataset, economy, center, normalize:
            (
                fn.pca,
                dt.ElMatrix.view_from_numpy(np.asarray(dataset, order='F')),
                dt.PcaControl.make(economy, center, normalize)
            )),
        ("ElDistMatrix", lambda dataset, economy, center, normalize:
            (
                fn.pca,
                dt.ElDistMatrix.make_view(
                    dt.ElGrid(MPI.COMM_WORLD),
                    dt.ElMatrix.view_from_numpy(np.asarray(dataset, order='F')),
                    dt.MatrixDistribution.make(
                        dt.ElDist.STAR,
                        dt.ElDist.STAR,
                        dt.ElDistWrap.ELEMENT)
                ),
                dt.PcaControl.make(economy, center, normalize)
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


@pytest.mark.parametrize("factory", TestArguments["factory"])
@pytest.mark.parametrize("dataset", TestArguments["dataset"])
@pytest.mark.parametrize("economy", TestArguments["economy"])
@pytest.mark.parametrize("center", TestArguments["center"])
@pytest.mark.parametrize("normalize", TestArguments["normalize"])
def test_fn_pca(env, factory, dataset, economy, center, normalize):
    factory_n = factory[0]
    factory_f = factory[1]
    logging.debug('factory:    %r' % factory_n)
    logging.debug('dataset:    %r' % type(dataset))
    logging.debug('economy:    %r' % economy)
    logging.debug('center:     %r' % center)
    logging.debug('normalize:  %r' % normalize)

    testcase = factory_f(dataset, economy, center, normalize)
    if testcase is None:
        pytest.skip("unsupported configuration")
        return

    logging.info("Rebuilding ~data~=score*coeff'+mean for impl %s" % factory_n)
    result = testcase[0](*testcase[1:])
    centered = fn.multiply(result.score, fn.transpose(result.coeff))
    rebuild = fn.plus(centered, fn.expand(result.mean, fn.size(centered))) if center else centered

    logging.info("Comparing original data and reconstructed data computed by impl %s" % factory_n)
    assert detail.test.matrix_matrix_allclose(dataset, rebuild)


@pytest.mark.parametrize("dataset", TestArguments["dataset"])
@pytest.mark.parametrize("economy", TestArguments["economy"])
@pytest.mark.parametrize("center", TestArguments["center"])
@pytest.mark.parametrize("normalize", TestArguments["normalize"])
def test_fn_pca_compare_results(env, dataset, economy, center, normalize):
    logging.debug('dataset:    %r' % type(dataset))
    logging.debug('economy:    %r' % economy)
    logging.debug('center:     %r' % center)
    logging.debug('normalize:  %r' % normalize)

    results = []
    factories = TestArguments["factory"]

    for factory in factories:
        factory_n = factory[0]
        factory_f = factory[1]
        logging.debug('factory:    %r' % factory_n)

        testcase = factory_f(dataset, economy, center, normalize)
        if testcase is None:
            logging.info("unsupported configuration")
            continue

        factory_result = testcase[0](*testcase[1:])
        results.append([factory_n, factory_f, testcase, factory_result])

    for i in range(0, len(results)-1):
        j = i+1

        factory_n_i, factory_f_i, testcase_i, factory_result_i = results[i]
        factory_n_j, factory_f_j, testcase_j, factory_result_j = results[j]
        logging.debug("comparing results for impl %s and %s" % (factory_n_i, factory_n_j))

        logging.info("comparing coeff of impl %s and %s" % (factory_n_i, factory_n_j))
        m, n = dataset.shape
        if (m < n) and not economy:
            rng = (
                dt.MatrixIndex.make(0, 0),
                dt.MatrixSize.make(n, m-1)  # TODO: Why n*DOF instead of n*m?
            )
            pca_coeff_i_mxn = fn.select(factory_result_i.coeff, rng)
            pca_coeff_j_mxn = fn.select(factory_result_j.coeff, rng)
            assert detail.test.matrix_matrix_allclose(pca_coeff_i_mxn, pca_coeff_j_mxn)
        else:
            assert detail.test.matrix_matrix_allclose(factory_result_i.coeff, factory_result_j.coeff)

        logging.info("comparing score of impl %s and %s" % (factory_n_i, factory_n_j))
        assert detail.test.matrix_matrix_allclose(factory_result_i.score,  factory_result_j.score)
        logging.info("comparing latent of impl %s and %s" % (factory_n_i, factory_n_j))
        assert detail.test.vector_vector_allclose(factory_result_i.latent, factory_result_j.latent)
        logging.info("comparing mean of impl %s and %s" % (factory_n_i, factory_n_j))
        assert detail.test.vector_vector_allclose(factory_result_i.mean, factory_result_j.mean)
        logging.info("comparing impl %s and %s done." % (factory_n_i, factory_n_j))
