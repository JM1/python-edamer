#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>

import edamer
import mpi4py.MPI
import pytest


def test_ctor():
    edamer.dt.ElGrid(mpi4py.MPI.Comm(mpi4py.MPI.COMM_WORLD))
    edamer.dt.ElGrid(mpi4py.MPI.COMM_WORLD)

    with pytest.raises(TypeError):
        edamer.dt.ElGrid("string_is_an_invalid_type")
