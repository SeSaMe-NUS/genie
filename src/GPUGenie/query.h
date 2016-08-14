/*! \file query.h
 *  \brief Declaration of query class
 *
 *  Query class is responsible for processing query for searching.
 *
 */

#ifndef GPUGenie_query_h
#define GPUGenie_query_h

#include "inv_table.h"
#include "inv_list.h"

#include <vector>

using namespace std;

namespace GPUGenie
{
/*! \typedef unsigned int u32
 */
typedef unsigned int u32;
/*! \typedef unsigned long long u64
 */
typedef unsigned long long u64;


/*! \class query
 *  \brief Query class includes the functions for processing queries based on user's input.
 *
 */
class query
{
public:
    /*! \struct range
     *  \brief The first-step struct for processing queries.
     *
     *  Every query would have a set of ranges. Every range would be transferred to
     *  struct dim later on.
     */
	struct range
	{
		int query;/*!< The query which the range belongs to */
        int order;/*!< Mainly used in subsequence match */
		int dim;/*!< The dimension which the range is on */
		int low;/*!< The starting value of this range */
		int up;/*!< The ending value of this range */
		float weight;/*!< Weight of this range */
		int low_offset;/*!< Offset on the lowerbound of this range */
		int up_offset;/*!< Offset on the upperbound of this range */
		float selectivity;/*!< The selectivity of this range */
	};
    /*! \struct dim
     *  \brief The second-step struct for processing queries.
     *
     */
    struct dim
    {
         int query;/*!< The query which the range belongs to */
         int order;/*!< Mainly used in subsequence search */
         int start_pos;/*!< The starting position in the ListArray for this dim */
         int end_pos;/*!< The ending position in the ListArray for this dim */
         float weight;/*!< Weight of this dim */
    };
private:
	/*! \var inv_table * _ref_table
	 *  \brief The refrenced table.
	 *
     *   The query can only be used to query the refrenced table.
	 */
	inv_table* _ref_table;

	/*! \var std::map<int, vector<range>*> _attr_map
	 *  \brief The saved ranges.
	 *
     *  The saved ranges. Since different table requires
	 *  different ranges setting, the attribute saves the
	 *  raw range settings.
	 */
	std::map<int, vector<range>*> _attr_map;

	/*! \var std::map<int, vector<dim>*> _dim_map
	 *  \brief The queried ranges.
	 *
     *   The queried ranges. In matching steps, this vector will
	 *   be transferred to device.
	 */
	std::map<int, vector<dim>*> _dim_map;

	/*! \var int _topk
	 *  \brief The top k matches required.
	 */
	int _topk;

    /*! \var float _selectivity
     *  \brief The selectivity for the set of queries.
     *
     *  The selectivity is used to expend the search range for a specific query set.
     *  So, we can get enough data points in the final result set.
     */
	float _selectivity;

    /*! \var int _index
     *  \brief Index of the query.
     */
	int _index;

    /*! \var int _count
     *  \brief number of dim
     */
	int _count;

public:
    /*! \var bool is_load_balanced
     *  \brief Whether this query set has been applied load balance feature.
     */
	bool is_load_balanced;
    /*! \var bool use_load_balance
     *  \brief Whether this query set is to use load balance.
     */
	bool use_load_balance;

    /*! \var vector<dim> _dims
     *  \brief Collection of dims of all queries
     */
	vector<dim> _dims;

	/*! \fn query(inv_table* ref, int index)
	 *  \brief Create a query based on an inv_table.
	 *
	 *  \param ref The refrence to the inv_table.
	 *  \param index The index for this query.
     */
	query(inv_table* ref, int index);

	/*! \fn query(inv_table& ref, int index)
	 *  \brief Create a query based on an inv_table.
	 *
	 *  \param ref The pointer to the inv_table.
	 *  \param index The index for this query.
     */
	query(inv_table& ref, int index);

    /*! \fn query()
     *  \brief Default empty constructor.
     */
	query();

	/*! \fn inv_table* ref_table()
	 *  \brief The refrenced table's pointer.
	 *  \return The pointer points to the refrenced table.
	 */
	inv_table*
	ref_table();

	/*! \fn void attr(int index, int low, int up, float weight )
	 *  \brief Modify the matching range and weight of an attribute.
	 *
	 *  \param index Attribute index
	 *  \param low Lower bound (included)
     *  \param up Upper bound (included)
	 *  \param weight The weight
	 */
	void
	attr(int index, int low, int up, float weight, int order);
	/*! \fn void attr(int index, int value, float weight, float selectivity)
	 *  \brief Set an attr struct.
	 *
	 *  \param index Attribute index
	 *  \param value Value of this attr
     *  \param weight Weight of this attr
	 *  \param selectivity The selectivity
	 */
	void
	attr(int index, int value, float weight, float selectivity, int order);
	/*! \fn void attr(int index, int low,int up ,float weight, float selectivity)
	 *  \brief Set an attr struct
	 *
	 *  \param index Attribute index
	 *  \param low Lower bound (included)
     *  \param up Upper bound (included)
     *  \param weight Weight of this attr
	 *  \param selectivity The selectivity
	 */
	void
	attr(int index, int low, int up, float weight, float selectivity, int order);

    /*! \fn void clear_dim(int index)
     *  \brief Delete the dim at index
     *
     *  \param index The index of the dim to be deleted.
     */
	void
	clear_dim(int index);

    /*! \fn void selectivity(float s)
     *  \brief Set the selectivity.
     *
     *  \param s Selectivity to set
     */
	void
	selectivity(float s);

    /*! \fn float selectivity()
     *  \brief The selectivity of the current settings
     *
     *  \return The current selectivity.
     */
	float
	selectivity();

    /*! \fn void apply_adaptive_query_range()
     *  \brief Construct query in adaptice range mode.
     *
     *  The function will construct adaptive query based the selectivity.
     *  Usually the range of the query would be expended after the function is called.
     *
     */
	void
	apply_adaptive_query_range();

    /*! \fn void build_and_apply_load_balance(int max_load)
     *  \brief The function would construct query in the setting where load balance feature is on
     *
     *  \param max_load  Max_load = posting_list_max_length * multiplier
     *
     *  The function is used when load balance feature is turned on. The inverted list is divided into
     *  sublists. This function is to re-locate query range on the inverted list, based on the sublists locations.
     *
     */
	void
	build_and_apply_load_balance(int max_load);

	/*! \fn void topk(int k)
	 *  \brief Set top k matches.
	 *
	 *  \param k The top k matches.
	 */
	void
	topk(int k);

	/*! \fn int topk()
	 *  \brief Get top k matches.
	 *
	 *  \return The top k matches.
	 */
	int
	topk();

	/*! \fn void build()
	 *  \brief Construct the query based on a normal table.
	 */
	void
	build();


	/*! \fn int dump(vector<dim>& vout)
	 *  \brief Transfer the matching information(the queried ranges) to vector vout.
	 *
	 *  \param vout The target vector.
     *
     *  Transfer matching information(the queried ranges) to vector vout.
	 *  vout will not be cleared, the push_back method will be invoked instead.
	 *
	 */
	int
	dump(vector<dim>& vout);

    /*! \fn void print(int limit)
     *  \brief Print out the information of all dims.
     *
     *  \param limit The maximum number of dims to print out.
     */
	void
	print(int limit);

    /*! \fn int index()
     *  \brief Get index of the query
     */
	int
	index();

    /*! \fn int count_ranges()
     *  \brief Get value of _ count
     *
     */
	int
	count_ranges();
};
}

#endif
