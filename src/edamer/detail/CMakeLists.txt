# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>
# vim:set fileformat=unix shiftwidth=4 softtabstop=4 expandtab:
# kate: end-of-line unix; space-indent on; indent-width 4; remove-trailing-spaces modified;
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

#################### build ####################

target_sources(py3 PRIVATE
    __init__.py)

#################### list the subdirectories ####################

add_subdirectory(log)
add_subdirectory(pybind11)
add_subdirectory(scalar)
add_subdirectory(test)
