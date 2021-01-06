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

#ifndef EDAMER_DETAIL_SCALAR_IMPL_HPP
#define EDAMER_DETAIL_SCALAR_IMPL_HPP

#include "fwd.hpp"

#include <boost/hana/drop_back.hpp>
#include <boost/hana/tuple.hpp>
#include <edamer/config.hpp>
#include <edamer/detail/pybind11.hpp>
#include <edamer/dt/el_complex.hpp>
#ifdef HBRS_MPL_ENABLE_ELEMENTAL
	#include <El.hpp>
#endif // HBRS_MPL_ENABLE_ELEMENTAL

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
EDAMER_NAMESPACE_BEGIN(detail)
namespace hana = boost::hana;

static auto scalars = hana::drop_back(hana::make_tuple(
	#ifdef EDAMER_ENABLE_SCALAR_INT
		EDAMER_TYPE_NAME_PAIR(int),
	#endif // EDAMER_ENABLE_SCALAR_INT

	#ifdef EDAMER_ENABLE_SCALAR_FLOAT
		EDAMER_TYPE_NAME_PAIR(float),
	#endif // EDAMER_ENABLE_SCALAR_FLOAT

	#ifdef EDAMER_ENABLE_SCALAR_DOUBLE
		EDAMER_TYPE_NAME_PAIR(double),
	#endif // EDAMER_ENABLE_SCALAR_DOUBLE

	"SEQUENCE_TERMINATOR___REMOVED_BY_DROP_BACK"
));

static auto complex_scalars = hana::drop_back(hana::make_tuple(
	#if defined(EDAMER_ENABLE_SCALAR_COMPLEX_FLOAT) && defined(HBRS_MPL_ENABLE_ELEMENTAL)
		EDAMER_TYPE_NAME_PAIR(El::Complex<float>),
	#endif // defined(EDAMER_ENABLE_SCALAR_COMPLEX_FLOAT) && defined(HBRS_MPL_ENABLE_ELEMENTAL)
	
	#if defined(EDAMER_ENABLE_SCALAR_COMPLEX_DOUBLE) && defined(HBRS_MPL_ENABLE_ELEMENTAL)
		EDAMER_TYPE_NAME_PAIR(El::Complex<double>),
	#endif // defined(EDAMER_ENABLE_SCALAR_COMPLEX_DOUBLE) && defined(HBRS_MPL_ENABLE_ELEMENTAL)

	"SEQUENCE_TERMINATOR___REMOVED_BY_DROP_BACK"
));

EDAMER_NAMESPACE_END(detail)

template <>
struct EDAMER_API pydef_impl<scalar_tag> {
	static py::module &
	apply(py::module & m, py::module & base);
};

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#endif // !EDAMER_DETAIL_SCALAR_IMPL_HPP
