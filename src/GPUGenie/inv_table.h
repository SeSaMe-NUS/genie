#ifndef GPUGenie_inv_table_h
#define GPUGenie_inv_table_h

#include <vector>
#include <map>
#include "inv_list.h"

typedef unsigned long long u64;

using namespace std;

namespace GPUGenie
{

class inv_table
{
public:
	enum status
	{
		not_builded, builded
	};
	//device pointer for some members
	int *d_inv_p;
	int *d_inv_index_p;
	int *d_inv_pos_p;
	int *d_ck_p;
	bool is_stored_in_gpu;
private:
    int table_index;//start at 0
    int total_num_of_table;//start at 1



    /**
	 * @brief Building status of the inv_table.
	 *        Any modification will make the
	 *        inv_table not_builded.
	 */
	status _build_status;

	/**
	 * @brief Bits shifted.
	 */
	int _shifter;

	/**
	 * @brief The number of instances.
	 */
	int _size;
    int _dim_size;
	/**
	 * @brief Inverted lists of different dimensions.
	 */
	vector<inv_list> _inv_lists;
    vector<int> inv_list_upperbound;
    vector<int> inv_list_lowerbound;

    vector<vector<int> > posting_list_size;



	/**
	 * @brief The composite keys' vector.
	 */
	vector<int> _ck;

	/**
	 * @brief The inverted indexes' vector.
	 */
	vector<int> _inv;
	/**
	 * @brief The first level index lists of posting lists vector
	 */
	vector<int> _inv_index;

	/**
	 * @brief The second level posting lists vector
	 */
	vector<int> _inv_pos;
	/**
	 * @brief The map used in compressed array.
	 */
	map<int, int> _ck_map;

public:

	/**
	 * @brief Default constructor of the inv_table.
	 * @details It sets the _shifter to 16 and set the
	 *        _size to -1.
	 */
	inv_table(): d_inv_p(NULL), d_inv_index_p(NULL), d_inv_pos_p(NULL), d_ck_p(NULL),
				is_stored_in_gpu(false),table_index(0),total_num_of_table(1),
                _build_status(not_builded), _shifter(16),_size(-1),_dim_size(0)
	{
	}

	~inv_table();


    static bool
    write(const char* filename, inv_table*& table);

    static bool
    read(const char* filename, inv_table*& table);

    void
    set_table_index(int index);

    void
    set_total_num_of_table(int num);

    int
    get_table_index();

    int
    get_total_num_of_table();


	bool
	cpy_data_to_gpu();

	void
	init();

	void
	clear_gpu_mem();

	/**
	 * @brief Clear the inv_table
	 * @details The method removes all content in _inv_lists, _ck and _inv.
	 *          It also sets the _size back to -1.
	 */
	void
	clear();

	/**
	 * @brief Check whether the inv_table is empty.
	 * @details If the _size is smaller than 0, the
	 *          method returns true and returns false
	 *          in the other case.
	 * @return true if _size < 0
	 *         false if _size >= 0
	 */
	bool
	empty();

	/**
	 * @brief Get the number of dimensions.
	 * @details Get the number of dimensions.
	 * @return The number of dimensions.
	 */
	int
	m_size();

	/**
	 * @brief Get the number of instances.
	 * @details Get the number of instances.
	 * @return The number of instances
	 */
	int
	i_size();

	/**
	 * @brief Get the shifter.
	 * @details Get the shifter.
	 * @return The shifter.
	 */
	int
	shifter();

	/**
	 * @brief Append an inv_list to the inv_table.
	 * @details The appended inv_list will be added
	 *          to _inv_lists. The first inv_list will set
	 *          the _size to correct number. If _size is not
	 *          equal to the size of inv_list or -1. This
	 *          method will simply return and do nothing.
	 * @param inv: the refrence to the inv_list which will be appended.
	 */
	void
	append(inv_list& inv);

	/**
	 * @brief Append an inv_list to the inv_table.
	 * @details The appended inv_list will be added
	 *          to _inv_lists. The first inv_list will set
	 *          the _size to correct number. If _size is not
	 *          equal to the size of inv_list or -1. This
	 *          method will simply return and do nothing.
	 * @param inv: the refrence to the inv_list which will be appended.
	 */
	void
	append(inv_list* inv);

	/**
	 * @brief Get building status of the inv_table.
	 * @details Get building status of the inv_table.
	 * @return Building status of the inv_table.
	 */
	status
	build_status();

	/**
	 * @brief The _inv_lists vector's pointer.
	 * @details The _inv_lists vector's pointer.
	 * @return The pointer points to _inv_lists vector.
	 */
	vector<inv_list>*
	inv_lists();

	/**
	 * @brief The _ck vector's pointer.
	 * @details The _ck vector's pointer
	 * @return The pointer points to _ck vector.
	 */
	vector<int>*
	ck();

	/**
	 * @brief The _inv vector's pointer.
	 * @details The _inv vector's pointer
	 * @return The pointer points to _inv vector.
	 */
	vector<int>*
	inv();
	vector<int>*
	inv_index();
	vector<int>*
	inv_pos();
	/**
	 * @brief The _ck_map's pointer.
	 * @details The _ck_map's pointer.
	 * @return The pointer points to _ck_map.
	 */
	map<int, int>*
	ck_map();


    //new feature upperboudn and lowerbound for each inv_list
    int
    get_upperbound_of_list(int index);

    int
    get_lowerbound_of_list(int index);




	/**
	 * @brief Build the inv_table.
	 * @details This method will merge all inv_lists to
	 *          two vector _ck and _inv and set the
	 *          _build_status to builded. Any query should
	 *          only be done after the inv_table has been builded.
	 */
	void
	build(u64 max_length, bool use_load_balance);


    //get length of each posting list
    int
    get_posting_list_size(int attr_index, int value);

    //contain certain value
    bool
    list_contain(int attr_index, int value);

    //for serialization
    bool
    write_to_file(ofstream& ofs);

    bool
    read_from_file(ifstream& ifs);






};
}


#endif
