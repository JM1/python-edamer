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

#include "impl.hpp"

#include <boost/format.hpp>
#include <boost/hana/at.hpp>
#include <boost/hana/cartesian_product.hpp>
#include <boost/hana/concat.hpp>
#include <boost/hana/first.hpp>
#include <boost/hana/flatten.hpp>
#include <boost/hana/fold_left.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/pair.hpp>
#include <boost/hana/transform.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/type.hpp>
#include <boost/hana/second.hpp>
#include <edamer/detail/scalar.hpp>
#include <edamer/dt/matrix_distribution.hpp>
#ifdef HBRS_MPL_ENABLE_ELEMENTAL
	#include <hbrs/mpl/dt/el_matrix.hpp>
	#include <hbrs/mpl/dt/el_vector.hpp>
	#include <hbrs/mpl/dt/el_dist_matrix.hpp>
	#include <hbrs/mpl/dt/el_dist_vector.hpp>
#endif // HBRS_MPL_ENABLE_ELEMENTAL
#include <hbrs/mpl/dt/pca_result/impl.hpp>
#include <memory>
#include <pybind11/numpy.h>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace hana = boost::hana;
namespace mpl = hbrs::mpl;

py::module &
pydef_impl<hbrs::mpl::pca_result_tag>::apply(py::module & m, py::module & base) {
	using hbrs::mpl::pca_result;
	using hbrs::mpl::pca_result_tag;
	
	auto py_pca_result = py::class_<pca_result_tag>{m, pystrip("pca_result").c_str()};
	
	auto ring_tns = hana::concat(detail::scalars, detail::complex_scalars);
	
	#ifdef HBRS_MPL_ENABLE_ELEMENTAL
	using hbrs::mpl::el_matrix;
	using hbrs::mpl::el_column_vector;
	using hbrs::mpl::el_row_vector;
	/* Example:
	 *   pca_result<
	 *     el_matrix<float>,
	 *     el_matrix<float>,
	 *     el_column_vector<float>,
	 *     el_row_vector<float>
	 *   >
	 */
	auto pca_result_el_matrix_tns = hana::transform(ring_tns, [](auto ring_tn){
		using ring_t = typename decltype(+hana::first(ring_tn))::type;
		auto ring_n = hana::second(ring_tn);
		
		using matrix_t = el_matrix<ring_t>;
		auto matrix_n = (boost::format("el_matrix<%s>") % ring_n).str();
		
		using column_vector_t = el_column_vector<ring_t>;
		auto column_vector_n = (boost::format("el_column_vector<%s>") % ring_n).str();
		
		using row_vector_t = el_row_vector<ring_t>;
		auto row_vector_n = (boost::format("el_row_vector<%s>") % ring_n).str();
		
		return hana::make_tuple(
			hana::make_pair(hana::type_c<matrix_t>, matrix_n),
			hana::make_pair(hana::type_c<matrix_t>, matrix_n),
			hana::make_pair(hana::type_c<column_vector_t>, column_vector_n),
			hana::make_pair(hana::type_c<row_vector_t>, row_vector_n)
		);
	});
	
	/* Example
	 *   pca_result<
	 *     el_dist_matrix<float, El::MC, El::MR, El::ELEMENT>,
	 *     el_dist_matrix<float, El::MC, El::MR, El::ELEMENT>,
	 *     el_dist_column_vector<float, El::MD, El::STAR, El::ELEMENT>,
	 *     el_dist_row_vector<float, El::STAR, El::VC, El::ELEMENT>
	 *   >
	 */
	using hbrs::mpl::el_dist_matrix;
	using hbrs::mpl::el_dist_column_vector;
	using hbrs::mpl::el_dist_row_vector;
	
	auto pca_result_el_dist_matrix_tns = hana::transform(
		hana::cartesian_product(hana::make_tuple(ring_tns, el_matrix_distributions)),
		[](auto product_tn) {
			auto ring_tn = hana::at_c<0>(product_tn);
			auto dist_tn = hana::at_c<1>(product_tn);
			
			using ring_t = typename decltype(+hana::first(ring_tn))::type;
			auto ring_n = hana::second(ring_tn);
			
			auto dist_ts = hana::transform(dist_tn, hana::first);
			auto dist_ns = hana::transform(dist_tn, hana::second);
			
			using columnwise_t = std::decay_t<decltype(hana::at_c<0>(dist_ts))>;
			auto columnwise_n  = hana::at_c<0>(dist_ns);
			using rowwise_t    = std::decay_t<decltype(hana::at_c<1>(dist_ts))>;
			auto rowwise_n     = hana::at_c<1>(dist_ns);
			using wrapping_t   = std::decay_t<decltype(hana::at_c<2>(dist_ts))>;
			auto wrapping_n    = hana::at_c<2>(dist_ns);
			
			using dist_matrix_t = el_dist_matrix<ring_t, columnwise_t::value, rowwise_t::value, wrapping_t::value>;
			auto dist_matrix_n = (boost::format{"el_dist_matrix<%s,%s,%s,%s>"}
				% ring_n % columnwise_n % rowwise_n % wrapping_n).str();
			
			using dist_column_vector_t = el_dist_column_vector<ring_t, El::MD, El::STAR, wrapping_t::value>;
			auto dist_column_vector_n = (boost::format{"el_dist_column_vector<%s,%s,%s,%s>"}
				% ring_n % "El::MD" % "El::STAR" % wrapping_n).str();
			
			using dist_row_vector_t = el_dist_row_vector<ring_t, El::STAR, El::VC, wrapping_t::value>;
			auto dist_row_vector_n = (boost::format{"el_dist_row_vector<%s,%s,%s,%s>"}
				% ring_n % "El::STAR" % "El::VC" % wrapping_n).str();
			
			return hana::make_tuple(
				hana::make_pair(hana::type_c<dist_matrix_t>, dist_matrix_n),
				hana::make_pair(hana::type_c<dist_matrix_t>, dist_matrix_n),
				hana::make_pair(hana::type_c<dist_column_vector_t>, dist_column_vector_n),
				hana::make_pair(hana::type_c<dist_row_vector_t>, dist_row_vector_n)
			);
	});
	
	auto el_pca_result_tns = hana::concat(pca_result_el_matrix_tns, pca_result_el_dist_matrix_tns);
	#endif // HBRS_MPL_ENABLE_ELEMENTAL
	
	hana::for_each(
		hana::flatten(hana::make_tuple(
			#ifdef HBRS_MPL_ENABLE_ELEMENTAL
			el_pca_result_tns
			#endif // HBRS_MPL_ENABLE_ELEMENTAL
		)),
		[&m, &py_pca_result](auto pairs) {
			auto types = hana::transform(pairs, hana::first);
			auto names = hana::transform(pairs, hana::second);
			auto name = boost::format("pca_result<%s>") %
				hana::fold_left(names, [](std::string s, std::string name) { return s + ',' + name; });
			
			using coeff_t = typename decltype(+hana::at_c<0>(types))::type;
			using score_t = typename decltype(+hana::at_c<1>(types))::type;
			using latent_t = typename decltype(+hana::at_c<2>(types))::type;
			using mean_t = typename decltype(+hana::at_c<3>(types))::type;
			
			using type_t = pca_result<coeff_t, score_t, latent_t, mean_t>;
			
			/* store template function pointers in variables to work around
			 * "unresolved overloaded function type" errors with GCC9/10
			 */
			auto c = py::class_<type_t>{m, pystrip(name.str()).c_str(), py_pca_result}
				.def(
					py::init<coeff_t, score_t, latent_t, mean_t>(),
					py::arg("coeff"),
					py::arg("score"),
					py::arg("latent"),
					py::arg("mean")
				);
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().coeff()), coeff_t &>) {
				c.def_property("coeff",
					[](type_t & o) { return o.coeff(); },
					[](type_t & o, coeff_t & v) { o.coeff() = HBRS_MPL_FWD(v); }
				);
			} else {
				c.def_property_readonly("coeff",
					[](type_t & o) { return o.coeff(); }
				);
			}
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().score()), score_t &>) {
				c.def_property("score",
					[](type_t & o) { return o.score(); },
					[](type_t & o, score_t & v) { o.score() = HBRS_MPL_FWD(v); }
				);
			} else {
				c.def_property_readonly("score",
					[](type_t & o) { return o.score(); }
				);
			}
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().latent()), latent_t &>) {
				c.def_property("latent",
					[](type_t & o) { return o.latent(); },
					[](type_t & o, latent_t & v) { o.latent() = HBRS_MPL_FWD(v); }
				);
			} else {
				c.def_property_readonly("latent",
					[](type_t & o) { return o.latent(); }
				);
			}
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().mean()), mean_t &>) {
				c.def_property("mean",
					[](type_t & o) { return o.mean(); },
					[](type_t & o, mean_t & v) { o.mean() = HBRS_MPL_FWD(v); }
				);
			} else {
				c.def_property_readonly("mean",
					[](type_t & o) { return o.mean(); }
				);
			}
		}
	);
	
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

