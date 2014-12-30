#ifndef GaLG_inv_table_h
#define GaLG_inv_table_h

#include "raw_data.h"
#include "inv_list.h"

#include <vector>

using namespace std;

namespace GaLG {
  class inv_table {
  private:
    /**
     * @brief Current status of the inv_table.
     *        Any modification will make the 
     *        inv_table not_builded.
     */
    enum {
      not_builded,
      builded
    } _build_status;

    /**
     * @brief Bits should be shifted.
     */
    int _shifter;

    /**
     * @brief The number of instances.
     */
    int _size;

    /**
     * @brief Inverted lists of different dimensions.
     */
    vector<inv_list> _inv_lists;

    /**
     * @brief The composite keys' vector.
     */
    vector<int> _ck;

    /**
     * @brief The inverted indexes' vector.
     */
    vector<int> _inv;

  public:
    /**
     * @brief Default constructor of the inv_table.
     * @details It set the _shifter to 16 and set the
     *        _size to -1.
     */
    inv_table() : _shifter(16), _size(-1) {}

    /**
     * @brief Clear the inv_table
     * @details The method remove all content in _inv_lists, _ck and _inv.
     *          It also sets the _size back to -1.
     */
    void clear();

    /**
     * @brief Check whether the inv_table is empty.
     * @details If the _size is smaller than 0, the
     *          method returns true and returns false
     *          in the other case.
     * @return true if _size < 0
     *         false if _size >= 0
     */
    bool empty();

    /**
     * @brief Get the number of dimensions.
     * @details Get the number of dimensions or meta datas.
     * @return The number of dimensions.
     */
    int m_size();

    /**
     * @brief Get the number of instances.
     * @details Get the number of instances or rows.
     * @return The number of instances
     */
    int i_size();

    /**
     * @brief Get the shifter.
     * @details Get the shifter.
     * @return The shifter.
     */
    int shifter();

    /**
     * @brief Append an inv_list to the inv_table.
     * @details The appended inv_list will be added
     *          to _inv_lists. The first inv_list will set
     *          the _size to correct number. If _size is not
     *          equal to the size of inv_list or -1. This
     *          method will simply return and do nothing.
     * @param inv: the refrence to the inv_list which will be appended.
     */ 
    void append(inv_list& inv);

    /**
     * @brief Build the inv_table.
     * @details This method will merge all inv_lists to
     *          two vector _ck and _inv and set the
     *          _build_status to builded. Any query should
     *          only be done after the inv_table has been builded.
     */
    void build();
  };
}

#endif