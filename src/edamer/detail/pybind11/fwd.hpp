/* Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef EDAMER_DETAIL_PYBIND11_FWD_HPP
#define EDAMER_DETAIL_PYBIND11_FWD_HPP

#include <boost/hana/integral_constant.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/type.hpp>
#include <boost/hana/pair.hpp>
#include <boost/preprocessor/seq/to_tuple.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <edamer/config.hpp>
#include <pybind11/pybind11.h>
#include <string>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace py = pybind11;

template <typename Tag, typename = void>
struct EDAMER_API pydef_impl;

struct pybind11_tag{};

template <>
struct pydef_impl<pybind11_tag>;

EDAMER_API
std::string
pystrip(std::string const& s);

EDAMER_API
std::string
regex_replace(std::string const& s, std::string const& re, std::string const& fmt);

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#define EDAMER_INTEGRAL_NAME_PAIR(integral)                                                                            \
	boost::hana::make_pair(                                                                                            \
		boost::hana::integral_c<std::decay_t<decltype(integral)>, integral>,                                           \
		std::string(BOOST_PP_STRINGIZE(integral))                                                                      \
	)

#define EDAMER_TYPE_NAME_PAIR(type)                                                                                    \
	boost::hana::make_pair(                                                                                            \
		boost::hana::type_c<type>,                                                                                     \
		std::string(BOOST_PP_STRINGIZE(type))                                                                          \
	)

#define _EDAMER_TYPE_NAME_PAIR_LAMBDA(r, data, elem)                                                                   \
	(EDAMER_TYPE_NAME_PAIR(elem))

#define EDAMER_TYPE_NAME_PAIRS(...)                                                                                    \
	boost::hana::make_tuple                                                                                            \
		BOOST_PP_SEQ_TO_TUPLE(                                                                                         \
			BOOST_PP_SEQ_FOR_EACH(_EDAMER_TYPE_NAME_PAIR_LAMBDA, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))             \
		)

#define EDAMER_DETAIL_PYBIND11_PYDEFS boost::hana::make_tuple(                                                         \
		edamer::pydef<edamer::pybind11_tag>                                                                            \
	)

#endif // !EDAMER_DETAIL_PYBIND11_FWD_HPP
