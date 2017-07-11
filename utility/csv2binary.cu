#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <genie/GPUGenie.h>
#include <genie/interface/io.h>
#include <genie/compression/DeviceCodecFactory.h>

using namespace genie;
using namespace genie::compression;
using namespace genie::table;
using namespace genie::utility;
using namespace std;

int main(int argc, char *argv[])
{
	namespace po = boost::program_options;
	
	std::string input_file, output_file, compression_type_str;

	po::options_description descriptions("Allowed options");
	descriptions.add_options()
		("help", "produce help message")
		("compression,c", po::value<std::string>(&compression_type_str)->default_value("no"), "set compression variant")
		("input,i", po::value<std::string>(&input_file), "input file")
		("output,o", po::value<std::string>(&output_file), "output file")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, descriptions), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		std::cout << descriptions << std::endl;
		return EXIT_SUCCESS;
	}

	if (!vm.count("input") || !vm.count("output"))
	{
		std::cout << "Input file or output file not provided!" << std::endl;
		return EXIT_FAILURE;
	}

	COMPRESSION_TYPE compression_type;
	if ("no" == compression_type_str)
		compression_type = NO_COMPRESSION;
	else if ("copy" == compression_type_str)
		compression_type = COPY;
	else if ("delta" == compression_type_str)
		compression_type = DELTA;
	else if ("bp32" == compression_type_str)
		compression_type = BP32;
	else if ("varint" == compression_type_str)
		compression_type = VARINT;
	else if ("comp-bp32-copy" == compression_type_str)
		compression_type = COMP_BP32_COPY;
	else if ("comp-bp32-varint" == compression_type_str)
		compression_type = COMP_BP32_VARINT;
	else if ("serial-copy-copy" == compression_type_str)
		compression_type = SERIAL_COPY_COPY;
	else if ("serial-delta-copy" == compression_type_str)
		compression_type = SERIAL_DELTA_COPY;
	else if ("serial-delta-delta" == compression_type_str)
		compression_type = SERIAL_DELTA_DELTA;
	else if ("serial-delta-varint" == compression_type_str)
		compression_type = SERIAL_DELTA_VARINT;
	else if ("serial-delta-bp32" == compression_type_str)
		compression_type = SERIAL_DELTA_BP32;
	else if ("serial-delta-comp-bp32-copy" == compression_type_str)
		compression_type = SERIAL_DELTA_COMP_BP32_COPY;
	else if ("serial-delta-comp-bp32-varint" == compression_type_str)
		compression_type = SERIAL_DELTA_COMP_BP32_VARINT;
	else
	{
		std::cout << "Invalid compression option" << std::endl;
		return EXIT_FAILURE;
	}

	GPUGenie_Config config;
	inv_table *table = nullptr;
	vector<vector<int> > data;
	config.data_points = &data;
	config.compression = compression_type;
	if (NO_COMPRESSION != compression_type)
		config.posting_list_max_length = 1024; // must set for compression

	read_file(data, input_file.c_str(), -1);
	preprocess_for_knn_csv(config, table);
	std::shared_ptr<const inv_table> sp_table(table, [](inv_table* ptr){delete[] ptr;});
	genie::SaveTableToBinary(output_file, sp_table);

	return EXIT_SUCCESS;
}
