#include <fstream>
#include <memory>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "serialization.h"

void
genie::util::SaveTable(const std::string &filename, const GPUGenie::inv_table* table)
{
    // Cannot save more than one table using this function
    if (table->get_table_index() != 0 || table->get_total_num_of_table() != 1)
        throw GPUGenie::genie_error("Saving multiple tables not supported");

    std::ofstream ofs(filename.c_str(), std::ios::binary | std::ios::trunc);
    boost::archive::text_oarchive oa(ofs);
    oa.register_type<GPUGenie::inv_compr_table>();
    oa << table;
}

std::shared_ptr<GPUGenie::inv_table>
genie::util::LoadTable(const std::string &filename)
{
    GPUGenie::inv_table *loaded_table = nullptr;
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    boost::archive::text_iarchive ia(ifs);
    ia.register_type<GPUGenie::inv_compr_table>();
    ia >> loaded_table;
    return shared_ptr<GPUGenie::inv_table>(loaded_table);
}

