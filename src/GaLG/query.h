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
    /**
     * @brief The refrenced table.
     * @details The refrenced table. The query can only be
     *        used to query the refrenced table.
     */
    inv_table* _ref_table;

    /**
     * @brief The saved ranges.
     * @details The saved ranges. Since different table requires
     *          different ranges setting, the attribute saves the
     *          raw range settings.
     */
    vector<dim> _attr;

    /**
     * @brief The queried ranges.
     * @details The queried ranges. In matching steps, this vector will
     *          be transferred to device.
     */
    vector<dim> _dims;

    /**
     * @brief The top k matches required.
     * @details The top k matches required.
     */
    int _topk;

  public:
    /**
     * @brief Create a query based on an inv_table.
     * @details Create a query based on an inv_table.
     * 
     * @param ref The refrence to the inv_table.
     */
    query(inv_table* ref);

    /**
     * @brief Create a query based on an inv_table.
     * @details Create a query based on an inv_table.
     * 
     * @param ref The pointer to the inv_table.
     */
    query(inv_table& ref);

    /**
     * @brief The refrenced table's pointer.
     * @details The refrenced table's pointer.
     * @return The pointer points to the refrenced table.
     */
    inv_table*
    ref_table();

    /**
     * @brief Modify the matching range and weight of an attribute.
     * @details Modify the matching range and weight of an attribute.
     * 
     * @param index Attribute index
     * @param low Lower bound (included)
     * @param up Upper bound (included)
     * @param weight The weight
     */
    void
    attr(int index, int low, int up, float weight);

    /**
     * @brief Set top k matches.
     * @details Set top k matches.
     *
     * @param k The top k matches.
     */
    void
    topk(int k);

    /**
     * @brief Get top k matches.
     * @details Get top k matches.
     *
     * @return The top k matches.
     */
    int
    topk();

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
     * @brief Transfer the matching information(the queried ranges) to vector vout.
     * @details Transfer matching information(the queried ranges) to vector vout.
     *          vout will not be cleared, the push_back method will be invoked instead.
     * 
     * @param vout The target vector.
     */
    void
    dump(vector<dim>& vout);
  };
}

#endif
