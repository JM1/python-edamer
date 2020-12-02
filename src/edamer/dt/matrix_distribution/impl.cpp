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

#include "impl.hpp"

#include <boost/container_hash/hash.hpp>
#include <boost/format.hpp>
#include <boost/hana/at.hpp>
#include <boost/hana/first.hpp>
#include <boost/hana/fold_left.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/integral_constant.hpp>
#include <boost/hana/transform.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/type.hpp>
#include <boost/hana/second.hpp>
#include <boost/throw_exception.hpp>
#include <edamer/dt/exception.hpp>
#include <hbrs/mpl/dt/matrix_distribution/impl.hpp>
#include <tuple>
#include <unordered_map>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace hana = boost::hana;
namespace mpl = hbrs::mpl;

EDAMER_NAMESPACE_BEGIN(/* unnamed */)

template<
	El::Dist Columnwise,
	El::Dist Rowwise,
	El::DistWrap Wrapping
>
py::object
make() {
	return py::cast(mpl::matrix_distribution<
		hana::integral_constant<El::Dist, Columnwise>,
		hana::integral_constant<El::Dist, Rowwise>,
		hana::integral_constant<El::DistWrap, Wrapping>
	>{});
}
EDAMER_NAMESPACE_END(/* unnamed */)

#ifdef HBRS_MPL_ENABLE_ELEMENTAL
EDAMER_NAMESPACE_BEGIN(detail)
py::list
el_matrix_distributions() {
	py::list list;
	hana::for_each(
		edamer::el_matrix_distributions,
		[&list](auto distribution_tn) {
			auto dist_ts = hana::transform(distribution_tn, hana::first);
			
			using columnwise_t = std::decay_t<decltype(hana::at_c<0>(dist_ts))>;
			using rowwise_t = std::decay_t<decltype(hana::at_c<1>(dist_ts))>;
			using wrapping_t = std::decay_t<decltype(hana::at_c<2>(dist_ts))>;
			
			list.append(make<columnwise_t::value, rowwise_t::value, wrapping_t::value>());
		}
	);
    return list;
}
EDAMER_NAMESPACE_END(detail)
#endif // HBRS_MPL_ENABLE_ELEMENTAL

py::module &
pydef_impl<hbrs::mpl::matrix_distribution_tag>::apply(py::module & m, py::module & base) {
	using hbrs::mpl::matrix_distribution;
	using hbrs::mpl::matrix_distribution_tag;
	auto py_matrix_distribution = py::class_<matrix_distribution_tag>{m, pystrip("matrix_distribution").c_str()};
	
	#ifdef HBRS_MPL_ENABLE_ELEMENTAL
	
	py::enum_<El::Dist>(m, pystrip("El::Dist").c_str())
		.value("MC",   El::MC,   "Col of a matrix distribution")
		.value("MD",   El::MD,   "Diagonal of a matrix distribution")
		.value("MR",   El::MR,   "Row of a matrix distribution")
		.value("VC",   El::VC,   "Col-major vector distribution")
		.value("VR",   El::VR,   "Row-major vector distribution")
		.value("STAR", El::STAR, "Give to every process")
		.value("CIRC", El::CIRC, "Give to a single process");
		
	py::enum_<El::DistWrap>(m, pystrip("El::DistWrap").c_str())
		.value("ELEMENT", El::ELEMENT)
		.value("BLOCK",   El::BLOCK);

	/* In Python, we want to select the matrix distribution at runtime, but in Elemental it is defined at
	 * compile-time using template arguments. Hence we add a list of predefined constructor functions (e.g. make) to a
	 * map, which can then be selected at runtime using a specific matrix distribution as map key.
	 */
	using make0_fun_t = std::function<py::object()>;
	using make0_key_t = std::tuple<El::Dist, El::Dist, El::DistWrap>;
	std::unordered_map<make0_key_t, make0_fun_t, boost::hash<make0_key_t>> make0_funs;

	hana::for_each(el_matrix_distributions, [&m, &make0_funs, &py_matrix_distribution](auto distribution_tn) {
		auto dist_ts = hana::transform(distribution_tn, hana::first);
		auto dist_ns = hana::transform(distribution_tn, hana::second);
		auto name = boost::format("matrix_distribution<%s>") %
			hana::fold_left(dist_ns, [](std::string s, std::string name) { return s + ',' + name; });

		using columnwise_t = std::decay_t<decltype(hana::at_c<0>(dist_ts))>;
		using rowwise_t = std::decay_t<decltype(hana::at_c<1>(dist_ts))>;
		using wrapping_t = std::decay_t<decltype(hana::at_c<2>(dist_ts))>;
		
		using matrix_distribution_t = matrix_distribution<
			hana::integral_constant<El::Dist, columnwise_t::value>,
			hana::integral_constant<El::Dist, rowwise_t::value>,
			hana::integral_constant<El::DistWrap, wrapping_t::value>
		>;
		
		/* store template function pointers in variables to work around
		* "unresolved overloaded function type" errors with GCC9/10
		*/
		constexpr auto make0_ptr = &make<columnwise_t::value, rowwise_t::value, wrapping_t::value>;
		
		make0_funs.emplace(std::make_tuple(columnwise_t::value, rowwise_t::value,wrapping_t::value), make0_ptr);
			
		py::class_<matrix_distribution_t>{m, pystrip(name.str()).c_str(), py_matrix_distribution}
			.def(py::init());
	});
	
	py_matrix_distribution.def_static("make",
		[make0_funs](
			El::Dist Columnwise,
			El::Dist Rowwise,
			El::DistWrap Wrapping
		) {
			make0_fun_t make0_fun;
			
			try {
				make0_fun = make0_funs.at(std::make_tuple(Columnwise, Rowwise, Wrapping));
			} catch (std::out_of_range&) {
				BOOST_THROW_EXCEPTION((matrix_distribution_not_supported_exception{}
					<< errinfo_el_matrix_distribution{{Columnwise, Rowwise, Wrapping}}
				));
			}
			
			return make0_fun();
		},
		py::arg("columnwise"),
		py::arg("rowwise"),
		py::arg("wrapping")
	);
	
	m.def("el_matrix_distributions", &detail::el_matrix_distributions);
	#endif // HBRS_MPL_ENABLE_ELEMENTAL
	
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)


