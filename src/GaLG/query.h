#ifndef GaLG_query_h
#define GaLG_query_h

#include "inv_table.h"
#include "inv_list.h"

#include <vector>

using namespace std;

namespace GaLG
{
  class query
  {
  public:
    struct dim
    {
      int low;
      int up;
      float weight;
    };

  private:
    inv_table* _ref_table;
    vector<dim> _attr;
    vector<dim> _dims;

  public:
    /**
     * @brief Create a query based on a inv_table.
     * @details Create a query based on a inv_table.
     * 
     * @param ref The refrence to the inv_table.
     */
    query(inv_table* ref);

    /**
     * @brief Create a query based on a inv_table.
     * @details Create a query based on a inv_table.
     * 
     * @param ref The pointer to the inv_table.
     */
    query(inv_table& ref);

    /**
     * @brief The refrence table's pointer.
     * @details The refrence table's pointer.
     * @return The pointer points to the refrence table.
     */
    inv_table*
    ref_table();

    /**
     * @brief Modify the matching range and weight of an attribute.
     * @details Modify the matching range and weight of an attribute.
     * 
     * @param index Attribute index
     * @param low Lower bound
     * @param up Upper bound
     * @param weight The weight
     */
    void
    attr(int index, int low, int up, float weight);

    /**
     * @brief Construct the query based on a normal table.
     * @details Construct the query based on a normal table.
     */
    void
    build();

    /**
     * @brief Construct the query based on a compressed table.
     * @details Construct the query based on a compressed table.
     */
    void
    build_compressed();

    /**
     * @brief Transfer matching information to vector vout.
     * @details Transfer matching information to vector vout. vout will not
     *          be cleared, the push_back method will be invoked instead.
     * 
     * @param vout The target vector.
     */
    void
    dump(vector<dim>& vout);
  };
}

#endif
