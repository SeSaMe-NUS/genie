/*
 * FileReader.h
 *
 *  Created on: Oct 26, 2015
 *      Author: zhoujingbo
 */

#ifndef GaLG_FILEREADER_H_
#define GaLG_FILEREADER_H_

#include <vector>
#include <string>

namespace GaLG {


std::vector<std::string> split(std::string& str, const char* c);
std::string eraseSpace(std::string origin) ;
void read_file(std::vector<std::vector<int> >& dest,
		        const char* fname,
		        int num);

} /* namespace GaLG */
#endif /* FILEREADER_H_ */
