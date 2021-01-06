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

#include <boost/regex.hpp>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)

EDAMER_API
std::string
/* convert C++ class names or C++ type names to valid Python names */
/* TODO: Rename pystring to e.g. pythonize? */
pystrip(std::string const& s) {
	/* std::regex does not support some Perl Format String features, such as \u, but Boost.Regex does.
	 * Ref.: https://www.boost.org/doc/libs/master/libs/regex/doc/html/boost_regex/format/perl_format.html
	 */

	// Remove underscores and convert lower case to title case
	static auto const re_types = boost::regex(R"([_]*([A-Za-z0-9]+))");
	std::string && title_case_types =
		boost::regex_replace(s, re_types, std::string{R"(\u$1)"});

	// Remove namespace markers (::)
	static auto const re_double_colons = boost::regex(R"(\:\:)");
	std::string && no_double_colons =
		boost::regex_replace(title_case_types, re_double_colons, std::string{R"(\u$1)"});
	// Replace template brackets (<>) with leading underscore
	
	// Remove template brackets (<>)
	static auto const re_angle_brackets = boost::regex(R"(<([^<>]*)>)");	
	std::string no_angle_brackets;
	for(std::string tmp = no_double_colons; no_angle_brackets != tmp;) {
		no_angle_brackets = tmp;
		tmp = boost::regex_replace(no_angle_brackets, re_angle_brackets, std::string{R"(_\u$1)"});
	}
	
	// remove invalid characters
	static auto const re_invalid_chars = boost::regex(R"([^A-Za-z0-9_])");
	std::string && valid_chars =
		boost::regex_replace(no_angle_brackets, re_invalid_chars, std::string{"_"});

	// remove leading and trailing underscores
	static auto const re_leading_trailing_underscores = boost::regex(R"((^_)|(_$))");
	std::string && no_leading_trailing_underscores =
		boost::regex_replace(valid_chars, re_leading_trailing_underscores, std::string{""});
	
	// remove multiple underscores
	static auto const re_multiple_underscores = boost::regex(R"(([_]+_))");
	return boost::regex_replace(no_leading_trailing_underscores, re_multiple_underscores, std::string{"_"});
}

EDAMER_API
std::string
regex_replace(std::string const& s, std::string const& re, std::string const& fmt) {
	return boost::regex_replace(s, boost::regex(re), fmt);
}

py::module &
pydef_impl<pybind11_tag>::apply(py::module & m, py::module & base) {
	m.def("pystrip", &pystrip, "convert C++ class names or C++ type names to valid Python names");
	m.def("regex_replace", &regex_replace, "Use regular expressions to perform substitutions on strings");
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)
