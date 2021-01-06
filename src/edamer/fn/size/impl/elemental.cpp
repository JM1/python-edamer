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

#include "elemental.hpp"

#ifdef HBRS_MPL_ENABLE_ELEMENTAL

#include "../fwd.hpp" // Make help(edamer.fn.*) show Python type names
#include <boost/hana/at.hpp>
#include <boost/hana/concat.hpp>
#include <boost/hana/first.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/type.hpp>
#include <boost/hana/second.hpp>
#include <edamer/detail/scalar.hpp>
#include <edamer/dt/matrix_distribution.hpp>
#include <hbrs/mpl/dt/el_dist_matrix.hpp>
#include <hbrs/mpl/dt/el_matrix.hpp>
#include <hbrs/mpl/dt/el_dist_vector.hpp>
#include <hbrs/mpl/dt/el_vector.hpp>
#include <hbrs/mpl/fn/size.hpp>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)

py::module &
pydef_impl<hbrs::mpl::detail::size_impl_el_matrix>::apply(py::module & m, py::module & base) {
	auto ring_tns = hana::concat(detail::scalars, detail::complex_scalars);
	
	using hbrs::mpl::el_matrix;
	
	hana::for_each(ring_tns, [&m](auto ring_tn) {
		using ring_t = typename decltype(+hana::first(ring_tn))::type;
		auto ring_n = hana::second(ring_tn);
		
		m.def("size",
			[](el_matrix<ring_t> const& a) {
				return hbrs::mpl::size(a);
			},
			py::arg("a")
		);
	});
	return m;
}

py::module &
pydef_impl<hbrs::mpl::detail::size_impl_el_dist_matrix>::apply(py::module & m, py::module & base) {
	auto ring_tns = hana::concat(detail::scalars, detail::complex_scalars);
	
	using hbrs::mpl::el_dist_matrix;
	
	hana::for_each(ring_tns, [&m](auto ring_tn) {
		using ring_t = typename decltype(+hana::first(ring_tn))::type;
		auto ring_n = hana::second(ring_tn);
		
		hana::for_each(el_matrix_distributions, [&](auto distribution_tn) {
			auto dist_ts = hana::transform(distribution_tn, hana::first);
			using columnwise_t = std::decay_t<decltype(hana::at_c<0>(dist_ts))>;
			using rowwise_t = std::decay_t<decltype(hana::at_c<1>(dist_ts))>;
			using wrapping_t = std::decay_t<decltype(hana::at_c<2>(dist_ts))>;
			
			m.def("size",
				[](el_dist_matrix<ring_t, columnwise_t::value, rowwise_t::value, wrapping_t::value> const& a) {
					return hbrs::mpl::size(a);
				},
				py::arg("a")
			);
		});
	});
	return m;
}

#define _EL_VECTOR(vector_kind)                                                                                        \
	py::module &                                                                                                       \
	pydef_impl<hbrs::mpl::detail::length_impl_el_ ## vector_kind ## _vector>::apply(                                   \
		py::module & m, py::module & base) {                                                                           \
		auto ring_tns = hana::concat(detail::scalars, detail::complex_scalars);                                        \
		                                                                                                               \
		using hbrs::mpl::el_ ## vector_kind ## _vector;                                                                \
		                                                                                                               \
		hana::for_each(ring_tns, [&m](auto ring_tn) {                                                                  \
			using ring_t = typename decltype(+hana::first(ring_tn))::type;                                             \
			auto ring_n = hana::second(ring_tn);                                                                       \
			                                                                                                           \
			m.def("size",                                                                                              \
				[](el_ ## vector_kind ## _vector<ring_t> const& v) {                                                   \
					return hbrs::mpl::size(v);                                                                         \
				},                                                                                                     \
				py::arg("v")                                                                                           \
			);                                                                                                         \
		});                                                                                                            \
		return m;                                                                                                      \
	}

_EL_VECTOR(column)
_EL_VECTOR(row)
#undef _EL_VECTOR

#define _EL_DIST_VECTOR(vector_kind)                                                                                   \
	py::module &                                                                                                       \
	pydef_impl<hbrs::mpl::detail::length_impl_el_dist_ ## vector_kind ## _vector>::apply(                              \
		py::module & m, py::module & base) {                                                                           \
		auto ring_tns = hana::concat(detail::scalars, detail::complex_scalars);                                        \
		                                                                                                               \
		using hbrs::mpl::el_dist_ ## vector_kind ## _vector;                                                           \
		                                                                                                               \
		hana::for_each(ring_tns, [&m](auto ring_tn) {                                                                  \
			using ring_t = typename decltype(+hana::first(ring_tn))::type;                                             \
			auto ring_n = hana::second(ring_tn);                                                                       \
			                                                                                                           \
			hana::for_each(el_matrix_distributions, [&](auto distribution_tn) {                                        \
				auto dist_ts = hana::transform(distribution_tn, hana::first);                                          \
				using columnwise_t = std::decay_t<decltype(hana::at_c<0>(dist_ts))>;                                   \
				using rowwise_t = std::decay_t<decltype(hana::at_c<1>(dist_ts))>;                                      \
				using wrapping_t = std::decay_t<decltype(hana::at_c<2>(dist_ts))>;                                     \
				                                                                                                       \
				m.def("size",                                                                                          \
					[](el_dist_ ## vector_kind ## _vector<                                                             \
						ring_t, columnwise_t::value, rowwise_t::value, wrapping_t::value                               \
					> const& v) {                                                                                      \
						return hbrs::mpl::size(v);                                                                     \
					},                                                                                                 \
					py::arg("v")                                                                                       \
				);                                                                                                     \
			});                                                                                                        \
		});                                                                                                            \
		return m;                                                                                                      \
	}

_EL_DIST_VECTOR(column)
_EL_DIST_VECTOR(row)
#undef _EL_DIST_VECTOR

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#endif // HBRS_MPL_ENABLE_ELEMENTAL
