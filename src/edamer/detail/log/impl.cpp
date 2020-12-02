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

#include <boost/log/expressions.hpp>
#include <hbrs/mpl/detail/log/impl.hpp>
#include <optional>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace mpl = hbrs::mpl;
namespace log = boost::log;

py::module &
pydef_impl<edamer::log_tag>::apply(py::module & m, py::module & base) {
	py::object logging = py::module::import("logging");
	py::object getLogger = logging.attr("getLogger");
	py::object logger = getLogger();
	py::object getEffectiveLevel = logger.attr("getEffectiveLevel");
	int LVL_CRITICAL = logging.attr("CRITICAL").cast<int>();
	int LVL_ERROR = logging.attr("ERROR").cast<int>();
	int LVL_WARNING = logging.attr("WARNING").cast<int>();
	int LVL_INFO = logging.attr("INFO").cast<int>();
	int LVL_DEBUG = logging.attr("DEBUG").cast<int>();
	int LVL_NOTSET = logging.attr("NOTSET").cast<int>();
	int lvl = getEffectiveLevel().cast<int>();
	
	auto translate = [LVL_CRITICAL, LVL_ERROR, LVL_WARNING, LVL_INFO, LVL_DEBUG, LVL_NOTSET]
		(int lvl) -> log::trivial::severity_level {
		if (lvl >= LVL_CRITICAL) {
			return log::trivial::fatal;
		} else if (lvl >= LVL_ERROR) {
			return log::trivial::error;
		} else if (lvl >= LVL_WARNING) {
			return log::trivial::warning;
		} else if (lvl >= LVL_INFO) {
			return log::trivial::info;
		} else if (lvl >= LVL_DEBUG) {
			return log::trivial::debug;
		} else if (lvl > LVL_NOTSET) {
			return log::trivial::trace;
		} else if (lvl == LVL_NOTSET) {
			/* "Note that the root logger is created with level WARNING."
			 * Ref.: https://docs.python.org/3/library/logging.html#logging.Logger.setLevel
			 */
			return log::trivial::warning;
		} else {
			// lvl < LVL_NOTSET
			return log::trivial::trace;
		}
	};
	
	auto set_level = [translate](int lvl) {
		log::core::get()->set_filter(
			log::trivial::severity >= translate(lvl)
		);
	};
	
	py::class_<log_tag>{m, pystrip("log").c_str()}
		.def_property_static("level",
			nullptr
			/* It is not possible to query boost::log::trivial::severity_level!
			 * Ref.: https://stackoverflow.com/questions/41144163/how-to-query-boostlog-severity/41144393#41144393
			 */,
			[set_level](py::object, int lvl) {
				set_level(lvl);
			});
	
	set_level(lvl); // Set Boost's log level to Python's current log level
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

