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

#include "elemental.hpp"

#ifdef HBRS_MPL_ENABLE_ELEMENTAL

#include "../fwd.hpp" // Make help(edamer.fn.*) show Python type names
#include <boost/hana/at.hpp>
#include <boost/hana/concat.hpp>
#include <boost/hana/cartesian_product.hpp>
#include <boost/hana/first.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/type.hpp>
#include <boost/hana/second.hpp>
#include <edamer/detail/scalar.hpp>
#include <edamer/dt/expression.hpp>
#include <edamer/dt/matrix_distribution.hpp>
#include <edamer/fn/expand.hpp> // Make help(edamer.fn.plus) show Python type names
#include <hbrs/mpl/dt/el_dist_matrix.hpp>
#include <hbrs/mpl/dt/el_matrix.hpp>
#include <hbrs/mpl/dt/expression.hpp>
#include <hbrs/mpl/fn/expand.hpp>
#include <hbrs/mpl/fn/plus.hpp>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace hana = boost::hana;
namespace mpl = hbrs::mpl;

py::module &
pydef_impl<hbrs::mpl::detail::plus_impl_el_matrix_el_matrix>::apply(py::module & m, py::module & base) {
	auto ring_tns = hana::concat(detail::scalars, detail::complex_scalars);
	
	using hbrs::mpl::el_matrix;
	
	hana::for_each(ring_tns, [&m](auto ring_tn) {
		using ring_t = typename decltype(+hana::first(ring_tn))::type;
		auto ring_n = hana::second(ring_tn);
		
		m.def("plus",
			[](el_matrix<ring_t> const& a, el_matrix<ring_t> const& b) {
				return hbrs::mpl::plus(a, b);
			},
			py::arg("a"),
			py::arg("b")
		);
		
		m.def("plus",
			[](el_matrix<ring_t> const& a, ring_t const& b) {
				return hbrs::mpl::plus(a, b);
			},
			py::arg("a"),
			py::arg("b")
		);
	});
	return m;
}

py::module &
pydef_impl<hbrs::mpl::detail::plus_impl_el_dist_matrix_expand_expr_el_dist_matrix>::\
apply(py::module & m, py::module & base) {
	auto ring_tns = hana::concat(detail::scalars, detail::complex_scalars);
	
	using namespace mpl;
	namespace hana = boost::hana;
	
	hana::for_each(ring_tns, [&m](auto ring_tn) {
		using ring_t = typename decltype(+hana::first(ring_tn))::type;
		auto ring_n = hana::second(ring_tn);
		
		hana::for_each(
			hana::cartesian_product(hana::make_tuple(el_matrix_distributions, el_matrix_distributions)),
			[&](auto product) {
				auto left_distribution_tn = hana::at_c<0>(product);
				auto right_distribution_tn = hana::at_c<1>(product);
				
				auto left_dist_ts = hana::transform(left_distribution_tn, hana::first);
				auto right_dist_ts = hana::transform(right_distribution_tn, hana::first);
				
				using left_columnwise_t = std::decay_t<decltype(hana::at_c<0>(left_dist_ts))>;
				using left_rowwise_t = std::decay_t<decltype(hana::at_c<1>(left_dist_ts))>;
				using left_wrapping_t = std::decay_t<decltype(hana::at_c<2>(left_dist_ts))>;
				
				using right_columnwise_t = std::decay_t<decltype(hana::at_c<0>(right_dist_ts))>;
				using right_rowwise_t = std::decay_t<decltype(hana::at_c<1>(right_dist_ts))>;
				using right_wrapping_t = std::decay_t<decltype(hana::at_c<2>(right_dist_ts))>;
				
				m.def("plus", [](
					el_dist_matrix<ring_t, left_columnwise_t::value, left_rowwise_t::value, left_wrapping_t::value>
						const& lhs,
					mpl::expression<
						expand_t,
						hana::tuple<
							el_dist_row_vector<
								ring_t, right_columnwise_t::value, right_rowwise_t::value, right_wrapping_t::value
							> const&,
							matrix_size<El::Int, El::Int> const&
						>
					> rhs
					) {
						return mpl::plus(lhs, rhs);
					},
					py::arg("lhs"),
					py::arg("rhs")
				);
			}
		);
	});
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#endif // HBRS_MPL_ENABLE_ELEMENTAL
