#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>
#
# EDAMER: Exascale Data Analysis Methods with Enhanced Reusability

import sys

if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is not supported.")

# list all submodules
from . import detail, dt, fn  # noqa 401
