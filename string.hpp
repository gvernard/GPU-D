#ifndef _OOC_STRING_H
#define _OOC_STRING_H

#include <string>
#include <sstream>

namespace mycuda {

	typedef std::string string;

	template <class T>
	inline string toString (const T &t){
		std::stringstream ss;
		ss << t;
		return ss.str();
	}
	
	template<typename T>
	inline bool fromString(const string &s, T &t){
		std::istringstream ss(s);
		ss >> t;
		if( ss.fail() ){
		  return false;
		} else {
		  return true;
		}
	}

} //namespace mycuda

#endif //_OOC_STRING_H
