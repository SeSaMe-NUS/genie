/** Name: test_6.cu
 * Description:
 * focus on serialization of table, especially under multiload setting
 *   sift data
 *   data is from csv file
 *   query is from csv file, single range
 *
 *
 */
#include <cassert>
#include <fstream>
#include <memory>

#include <genie/original/interface.h>
#include <genie/table/inv_table.h>
#include <genie/table/inv_compr_table.h>
#include <genie/interface/io.h>

using namespace genie::compression;
using namespace genie::table;
using namespace genie::utility;
using namespace std;

void testSerialization(genie::original::GPUGenie_Config &config, const std::string inv_filename)
{
    Logger::log(Logger::INFO, "Preprocessing inverted table...");
    inv_table * table = nullptr;
    preprocess_for_knn_csv(config, table); // this returns inv_compr_table if config.compression is set
    assert(table != nullptr);
    assert(table->build_status() == inv_table::builded);
    shared_ptr<inv_table> sp_table(table, std::default_delete<inv_table[]>());

    Logger::log(Logger::INFO, "Saving inverted table to file...");
    genie::SaveTableToBinary(inv_filename, sp_table);

    Logger::log(Logger::INFO, "Loading inverted table from file...");
    std::shared_ptr<inv_table> loaded_table = genie::LoadTableFromBinary(inv_filename);

    Logger::log(Logger::INFO, "Checking loaded table correctness...");

    assert(loaded_table->d_inv_p == nullptr);
    assert(loaded_table->is_stored_in_gpu == false);
    assert(loaded_table->get_table_index() == 0);
    assert(loaded_table->get_total_num_of_table() == 1);

    assert(table->shift_bits_sequence == loaded_table->shift_bits_sequence);
    assert(table->m_size() == loaded_table->m_size());
    assert(table->i_size() == loaded_table->i_size());
    assert(table->shifter() == loaded_table->shifter());
    assert(table->_shift_bits_subsequence() == loaded_table->_shift_bits_subsequence());
    assert(table->build_status() == loaded_table->build_status());
    assert(*table->inv() == *loaded_table->inv());
    assert(*table->inv_pos() == *loaded_table->inv_pos());
    assert(*table->inv_index_map() == *loaded_table->inv_index_map());
    assert(table->get_min_value_sequence() == loaded_table->get_min_value_sequence());
    assert(table->get_max_value_sequence() == loaded_table->get_max_value_sequence());
    assert(table->get_gram_length_sequence() == loaded_table->get_gram_length_sequence());

    // Private variables of inv_table not checked by this test:
    //     _distinct_map
    //     inv_list_lowerbound
    //     inv_list_upperbound
    //     posting_list_size

    if (config.compression)
    {
        inv_compr_table *ctable = dynamic_cast<inv_compr_table*>(table);
        inv_compr_table *c_loaded_table = dynamic_cast<inv_compr_table*>(loaded_table.get());
        assert(ctable);
        assert(c_loaded_table);
        assert(*ctable->compressedInv() == *c_loaded_table->compressedInv());
        assert(*ctable->uncompressedInv() == *c_loaded_table->uncompressedInv());
        assert(*ctable->compressedInvPos() == *c_loaded_table->compressedInvPos());
        assert(*ctable->uncompressedInvPos() == *c_loaded_table->uncompressedInvPos());
        assert(ctable->getUncompressedPostingListMaxLength() == c_loaded_table->getUncompressedPostingListMaxLength());
        assert(ctable->getCompression() == c_loaded_table->getCompression());
    }
}


int main(int argc, char* argv[])
{
    string dataFile = "../static/sift_20.csv";

    Logger::set_level(Logger::DEBUG);
    Logger::log(Logger::INFO, "Reading csv data file %s ...", dataFile.c_str());
    std::vector<std::vector<int>> data;
    genie::original::GPUGenie_Config config;
    config.data_points = &data;
    config.data_type = 0;
    config.posting_list_max_length = 1024;
    read_file(data, dataFile.c_str(), -1);

    // Test inv_table
    config.compression = NO_COMPRESSION;
    testSerialization(config,"/tmp/genie_test_serialization.invtable");

    // Test inv_compr_table
    config.compression = HEAVYWEIGHT_COMPRESSION_TYPE;
    testSerialization(config,"/tmp/genie_test_serialization.cinvtable");

    return 0;
}
