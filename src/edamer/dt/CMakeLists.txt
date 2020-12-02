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

add_subdirectory(el_dist_matrix)
add_subdirectory(el_dist_vector)
add_subdirectory(el_grid)
add_subdirectory(el_matrix)
add_subdirectory(el_vector)
add_subdirectory(exception)
add_subdirectory(expression)
add_subdirectory(matrix_distribution)
add_subdirectory(matrix_index)
add_subdirectory(matrix_size)
add_subdirectory(pca_control)
add_subdirectory(pca_result)
add_subdirectory(range)
