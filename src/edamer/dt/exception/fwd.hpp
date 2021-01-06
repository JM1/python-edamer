/* Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef EDAMER_DT_EXCEPTION_FWD_HPP
#define EDAMER_DT_EXCEPTION_FWD_HPP

#include <boost/exception/error_info.hpp>
#include <boost/hana/tuple.hpp>
#include <edamer/config.hpp>
#include <edamer/detail/pybind11.hpp>
#include <hbrs/mpl/config.hpp>
#include <tuple>

#ifdef HBRS_MPL_ENABLE_ELEMENTAL
	#include <El.hpp>
#endif // HBRS_MPL_ENABLE_ELEMENTAL

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)

struct exception_tag{};

template <>
struct pydef_impl<exception_tag>;

struct EDAMER_API incompatible_ndarray_exception;
struct EDAMER_API import_mpi4py_failed_exception;
struct EDAMER_API matrix_distribution_not_supported_exception;

typedef boost::error_info<struct errinfo_ndarray_ndim_, py::ssize_t>
	errinfo_ndarray_ndim;

#ifdef HBRS_MPL_ENABLE_ELEMENTAL
typedef boost::error_info<struct errinfo_el_matrix_distribution_, std::tuple<El::Dist, El::Dist, El::DistWrap>>
	errinfo_el_matrix_distribution;
#endif // HBRS_MPL_ENABLE_ELEMENTAL

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#define EDAMER_DT_EXCEPTION_PYDEFS boost::hana::make_tuple(                                                            \
		edamer::pydef<edamer::exception_tag>                                                                           \
	)

#endif // !EDAMER_DT_EXCEPTION_FWD_HPP
