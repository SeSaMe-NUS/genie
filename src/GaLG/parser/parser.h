#ifndef GaLG_parser_h
#define GaLG_parser_h

#include "../raw_data.h"//for ide: to revert it as system file later, change "GaLG/raw_data.h" to "../raw_data.h"

namespace GaLG
{
  namespace parser
  {
    /**
     * @brief Parsing the csv file.
     * @details Reading data from the csv file and
     *          store the information as a raw_data
     *          instance.
     * 
     * @param file The path to the csv file.
     * @param data The reference to the raw_data instance.
     *             The data in this instance will be replaced
     *             by the data read from the csv file.
     * 
     * @return Lines of csv file.
     */
    int
    csv(string file, raw_data& data);

    /**
     * @brief Parsing the csv file.
     * @details Reading data from the csv file and
     *          store the information as a raw_data
     *          instance.
     *
     * @param file The path to the csv file.
     * @param data The reference to the raw_data instance.
     *             The data in this instance will be replaced
     *             by the data read from the csv file.
     * @param include_meta Include the meta data or not.
     *
     * @return Lines of csv file.
     */
    int
    csv(string file, raw_data& data, bool include_meta);
  }
}

#endif
