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

#ifndef EDAMER_DETAIL_PYBIND11_IMPL_HPP
#define EDAMER_DETAIL_PYBIND11_IMPL_HPP

#include "fwd.hpp"

#include <boost/format.hpp>
#include <boost/hana/tuple.hpp>
#include <edamer/dt/expression.hpp>
#include <hbrs/mpl/dt/expression.hpp>
#include <pybind11/pybind11.h>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace hana = boost::hana;
namespace mpl = hbrs::mpl;

template <typename Tag>
struct pydef_t {
	py::module &
	operator()(py::module & m, py::module & base) const {
		return pydef_impl<Tag>::apply(m, base);
	}
};

template <typename Tag>
constexpr pydef_t<Tag> pydef{};

template <>
struct EDAMER_API pydef_impl<pybind11_tag> {
	static py::module &
	apply(py::module & m, py::module & base);
};

template <typename Operation, typename... Operands>
EDAMER_API
py::module &
pydef_expression(
	py::module & base,
	hana::basic_type<mpl::expression<Operation, hana::tuple<Operands...>>>
) {
	auto m_dt = base.attr("dt").cast<py::module_>();
	auto py_expression = m_dt.attr(pystrip("expression").c_str());
	
	using operation_t = Operation;
	using operands_t = hana::tuple<Operands...>;
	
	std::string operation_n = py::detail::make_caster<operation_t>::name.text;
	/* We cannot use e.g.
	 *     py::detail::get_type_info(typeid(operation_t), / * throw_if_missing * / true)->type->tp_name
	 * because operation_t is not registered as a type but transparently casted between C++ and Python using 
	 * a pybind11::detail::type_caster<> template specialization
	 */
	std::string operands_n = typeid(hana::tuple<operands_t>).name();
	/* NOTE: We have to use this cryptic name from typeid() here because we have to store the complete type info of
	 * operands including constness and references. If we dismiss cv-qualifiers and references, then we run into
	 * undefined behaviour because pybind11 cast e.g. from "el_matrix<...> const&" to "el_matrix<...>" etc.
	 *
	 * This code is WRONG as it will strip constness and references:
	 *   #include <boost/hana/fold_left.hpp>
	 *
	 *   static const auto split_class = [](std::string c) {
	 *       auto idx = c.find_last_of('.');
	 *       if (std::string::npos == idx) {
	 *           return c;
	 *       } else {
	 *           return c.substr(idx + 1);
	 *       }
	 *   };
	 *   std::string operands_n2 = hana::fold_left(
	 *       hana::make_tuple(
	 *           std::string{
	 *               py::detail::get_type_info(typeid(Operands), / * throw_if_missing * / true)->type->tp_name
	 *           }...
	 *       ),
	 *       [](std::string lhs, std::string rhs) {
	 *         return split_class(lhs) + ',' + split_class(rhs);
	 *       }
	 *   );
	 */
	
	auto name = boost::format("expression<%s,%s>") % operation_n % operands_n;
	using type_t = mpl::expression<operation_t, operands_t>;
	
	if (py::detail::get_type_info(typeid(type_t), /* throw_if_missing */ false) != nullptr) {
		// C++ type registered already
		return base;
	}
	/* TODO: Investigate why Python fails to 'import edamer.cpp' with error message "ImportError: UnicodeDecodeError:
	 * 'utf-8' codec can't decode byte 0xb0 in position 0: invalid start byte" if CMAKE_BUILD_TYPE=Release and class 'c'
	 * inherits from py_expression. This error occurs both with GCC10 and Clang10 and might have to do with pybind11 and
	 * Link Time Optimization (LTO).
	 */
	auto c = py::class_<type_t>{m_dt, pystrip(name.str()).c_str()/*, py_expression*/}
		.def(
			py::init(
				[](operation_t o, operands_t os) -> type_t {
					return {HBRS_MPL_FWD(o), HBRS_MPL_FWD(os)};
				}
			),
			py::arg("operation"),
			py::arg("operands")
		);

	if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().operation()), operation_t>) {
		c.def_property("operation",
			[](type_t & o) { return o.operation(); },
			[](type_t & o, operation_t & v) { o.operation() = HBRS_MPL_FWD(v); }
		);
	} else {
		c.def_property_readonly("operation",
			[](type_t & o) { return o.operation(); }
		);
	}

	//TODO: Implement pybind11::detail::type_caster for boost::hana::tuple
	if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().operands()), operands_t>) {
		c.def_property("operands",
			[](type_t & o) { return o.operands(); },
			[](type_t & o, operands_t & v) { o.operands() = HBRS_MPL_FWD(v); }
		);
	} else {
		c.def_property_readonly("operands",
			[](type_t & o) { return o.operands(); }
		);
	}

	return base;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#endif // !EDAMER_DETAIL_PYBIND11_IMPL_HPP
