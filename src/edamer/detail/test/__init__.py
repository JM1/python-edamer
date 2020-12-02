#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>

from edamer import dt
import numpy as np
import logging


def to_numpy_1d(a):
    if isinstance(a, np.ndarray) and a.ndim == 1:
        return a
    elif isinstance(a, dt.ElColumnVector) or isinstance(a, dt.ElRowVector):
        return a.view_to_numpy()
    elif isinstance(a, dt.ElDistColumnVector) or isinstance(a, dt.ElDistRowVector):
        dist_star_star_el = dt.MatrixDistribution.make(dt.ElDist.STAR, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)
        return a.copy(dist_star_star_el).local().view_to_numpy()
    else:
        raise NotImplementedError("%s is not supported" % type(a))


def to_numpy_2d(a):
    if isinstance(a, np.ndarray) and a.ndim == 2:
        return a
    elif isinstance(a, dt.ElMatrix):
        return a.view_to_numpy()
    elif isinstance(a, dt.ElDistMatrix):
        dist_star_star_el = dt.MatrixDistribution.make(dt.ElDist.STAR, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)
        return a.copy(dist_star_star_el).local().view_to_numpy()
    else:
        raise NotImplementedError("%s is not supported" % type(a))


def vector_vector_allclose(a, b, rtol=1e-05, atol=1e-08):
    a_np = to_numpy_1d(a)
    b_np = to_numpy_1d(b)
    if np.allclose(a_np, b_np, rtol, atol):
        return True

    logging.info("a: %r" % a_np)
    logging.info("b: %r" % b_np)
    return False


def matrix_matrix_allclose(a, b, rtol=1e-05, atol=1e-08):
    a_np = to_numpy_2d(a)
    b_np = to_numpy_2d(b)
    if np.allclose(a_np, a_np, rtol, atol):
        return True

    logging.info("a: %r" % a_np)
    logging.info("b: %r" % b_np)
    return False


def vector_vector_equal(a, b):
    a_np = to_numpy_1d(a)
    b_np = to_numpy_1d(b)
    if np.array_equal(a_np, b_np):
        return True

    logging.info("a: %r" % a_np)
    logging.info("b: %r" % b_np)
    return False


def matrix_matrix_equal(a, b):
    a_np = to_numpy_2d(a)
    b_np = to_numpy_2d(b)
    if np.array_equal(a_np, b_np):
        return True

    logging.info("a: %r" % a_np)
    logging.info("b: %r" % b_np)
    return False
