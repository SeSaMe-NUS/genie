/** Name: test_6.cu
 * Description:
 * focus on serialization of table, especially under multiload setting
 *   sift data
 *   data is from csv file
 *   query is from csv file, single range
 *
 *
 */
#pragma warning(disable:4099)

#include <cassert>
#include <fstream>
#include <memory>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <GPUGenie/interface.h>
#include <GPUGenie/inv_table.h>
#include <GPUGenie/inv_compr_table.h>
#include <GPUGenie/serialization.h>

using namespace std;
using namespace GPUGenie;


void testSimpleSerialization(GPUGenie::GPUGenie_Config &config, const std::string inv_filename)
{
    Logger::log(Logger::INFO, "Preprocessing inverted table...");
    GPUGenie::inv_table * table = nullptr;
    GPUGenie::preprocess_for_knn_csv(config, table); // this returns inv_compr_table if config.compression is set
    assert(table != nullptr);
    assert(table->build_status() == inv_table::builded);

    Logger::log(Logger::INFO, "Saving inverted table to file...");
    genie::util::SaveTable(inv_filename, table);

    Logger::log(Logger::INFO, "Loading inverted table from file...");
    std::shared_ptr<GPUGenie::inv_table> loaded_table = genie::util::LoadTable(inv_filename);

    Logger::log(Logger::INFO, "Checking loaded table correctness...");

    // assert(table->table_index == loaded_table->table_index);

    if (config.compression)
    {
        GPUGenie::inv_compr_table *ctable = dynamic_cast<GPUGenie::inv_compr_table*>(loaded_table.get());
        assert(ctable);
    }


    Logger::log(Logger::DEBUG, "Deallocating inverted table...");
    delete[] table;
}

/**
 * Run a test of inverted table serialization using custom boost arhive class, i.e. create new template instance of the
 * serialization functions of the inverted table
 */
void testCustomSerialization(GPUGenie::GPUGenie_Config &config, const std::string inv_filename)
{
    Logger::log(Logger::INFO, "Preprocessing inverted table...");
    GPUGenie::inv_table * table = nullptr;
    GPUGenie::preprocess_for_knn_csv(config, table); // this returns inv_compr_table if config.compression is set
    assert(table != nullptr);
    assert(table->build_status() == inv_table::builded);
    
    Logger::log(Logger::INFO, "Saving inverted table to file...");
    {
        std::ofstream ofs(inv_filename.c_str());
        boost::archive::text_oarchive oa(ofs);
        oa << *table;
    }
    
    GPUGenie::inv_table * loaded_table = new inv_table();
    Logger::log(Logger::INFO, "Loading inverted table from file...");
    {
        std::ifstream ifs(inv_filename.c_str());
        boost::archive::text_iarchive ia(ifs);
        ia >> *loaded_table;
    }

    Logger::log(Logger::INFO, "Checking loaded table correctness...");

    // assert(table->table_index == loaded_table->table_index);

    Logger::log(Logger::DEBUG, "Deallocating inverted table...");
    delete[] table;
    delete loaded_table;

}

int main(int argc, char* argv[])
{
    string dataFile = "../static/sift_20.csv";

    Logger::set_level(Logger::DEBUG);
    Logger::log(Logger::INFO, "Reading csv data file %s ...", dataFile.c_str());
    std::vector<std::vector<int>> data;
    GPUGenie::GPUGenie_Config config;
    config.data_points = &data;
    config.data_type = 0;
    config.posting_list_max_length = 1024;
    GPUGenie::read_file(data, dataFile.c_str(), -1);

    // Test inv_table
    config.compression = NO_COMPRESSION;
    testSimpleSerialization(config,"/tmp/genie_test_serialization.invtable");

    // Test inv_compr_table
    config.compression = HEAVYWEIGHT_COMPRESSION_TYPE;
    testSimpleSerialization(config,"/tmp/genie_test_serialization.cinvtable");

    // Test inv_table
    config.compression = NO_COMPRESSION;
    testCustomSerialization(config,"/tmp/genie_test_serialization.invtable.txt");

    // Test inv_compr_table
    config.compression = HEAVYWEIGHT_COMPRESSION_TYPE;
    testCustomSerialization(config,"/tmp/genie_test_serialization.cinvtable.txt");

    return 0;
}
