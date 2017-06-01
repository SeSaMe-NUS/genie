/*! \file inv_table.h
 *  \brief define class inv_table
 *
 *  This file contains the declaration for inv_table class
 */



#ifndef GPUGenie_inv_table_h
#define GPUGenie_inv_table_h

#include <vector>
#include <map>
#include <unordered_map>
#include "inv_list.h"

/*! \var typedef unsigned long long u64
 *  \brief A type definition for a 64-bit unsigned integer
 */
typedef unsigned long long u64;

using namespace std;

/*! \namespace GPUGenie
 *  \brief GPUGenie is the top namespace for the project
 */
namespace GPUGenie
{


/*! \class inv_table
 *  \brief The declaration for class inv_table
 *
 *  The inv_table class includes the inverted index structure for a specific dataset
 *  to be searched. Also this class contains information for constructing query on this set.
 *  In one word, this class manages all information about the inverted index.
 */

class inv_table
{
public:

    /*! \enum status
     *  \brief This enum var defines two statuses for a inv_table object, which is either builded or not_builded.
     */
	enum status
	{
		not_builded, builded
	};
    /*! \var int *d_inv_p
     *  \brief d_inv_p points to the start location for posting list array in GPU memory.
     */
	int *d_inv_p = NULL;

    /*! \var static int max_inv_Size
     *  \brief This variable is used in multi-load mode. Yhe device pointer is repeatedly used, thus it should be able to contain maximum
     *  inverted list among all inv_tables.
     */
    static int max_inv_size;

    /*! \var bool is_stored_in_gpu
     *  \brief is_stored_in_gpu tell whether inverted index structure is pre-stored inside gpu memory
     *
     *  If true, data is pre-stored inside gpu so that you can launch queries multiple times. only need to
     *  transfer data to device once. If false, you have to transfer data to device every time you launch a query.
     *  (Here, one query means one execution of search)
     */
	bool is_stored_in_gpu;

    /*! \var int shift_bits_sequence
     *  \brief This variable is used to tell the number of bits shifted for recording gram in different position.
     */
    int shift_bits_sequence;



private:

    /*! \var int table_index
     *  \brief table_index is useful only when you want to serialize the obeject to a binary file and in a mult-load mode.
     *
     *  It specifies the index of this table in the table array for a whole data set, which is too big to be included in only
     *  one inv_table. The index starts at 0.
     */
    int table_index;

    /*! \var int total_num_of_table
     *  \brief total_num_of_table tells the number of inv_table in the table array for a dataset.
     *
     *  It is 1, when there is only one table. It is more than 1, if multi-load mode is used. In this mode
     *  multiple tables will be built sequentially for one dataset due to the large size of the dataset.
     */
    int total_num_of_table;



    /*! \var status _build_status
     *  \brief Building status of the inv_table.
     *        Any modification will make the
     *        inv_table not_builded.
     */
    status _build_status;

    /*! \var int _shifter
     *  \brief Bits shifted.
     *
     */
    int _shifter;

    /*! \var int _size
     *  \brief The number of instances.
     */
    int _size;

    /*! \var int _dim_size
     *  \brief The number of dim for the corresponding dataset.
     */
    int _dim_size;

    /*! \var vector<inv_list> _inv_lists;
     *  \brief Inverted lists of different dimensions.
     */
    vector<inv_list> _inv_lists;

    /*! \var vector<vector<int> > distinct_value
     *  \brief distinct_value for each sequence;
     *
     */
    //vector<vector<int> > distinct_value;
    vector<unordered_map<int, int> > _distinct_map;


    /*! \var vector<int> inv_list_upperbound
     *  \brief the maximum value for one inv_list. The id is implicitly expressed by the vector index.
     */
    vector<int> inv_list_upperbound;

    /*! \var vector<int> inv_list_lowerbound
     *  \brief the minimum value for one inv_list. The id is implicitly expressed by the vector index.
     */
    vector<int> inv_list_lowerbound;

    /*! \var vector<vector<int> > posting_list_size
     *  \brief Every attribute and value would correspond a posting list in inverted index. This vector
     *  contains length of every posting list, referred by one attribute-value pair.
     */
    vector<vector<int> > posting_list_size;



    /*! \var vector<int> _ck
     *  \brief The composite keys' vector.
     */
     vector<int> _ck;

    /*! \var vector<int> _inv
     *  \brief The inverted indexes' vector.
     *
     *  _inv contains the all the posting list which consist of ids of data points. The d_inv_p points to
     *  the corresponding memory on GPU.
     */
     vector<int> _inv;

     /*! \var vector<int> _inv_index
      *  \brief The first level index lists of posting lists vector
      */
     vector<int> _inv_index;
	unordered_map<size_t, int> _inv_index_map;

     /*! \var vector<int> _inv_pos
      *  \brief The second level posting lists vector
      */
      vector<int> _inv_pos;

      /*! \var unsigned int shift_bits_subsequence
       *  \brief The number of shifted bits in subsequence search.
       *
       *
       */
      unsigned int shift_bits_subsequence;

      /*! \var int min_value_sequence
       *  \brief The min value of all sequences that are kept in this inv_table object
       */
      int min_value_sequence;


      /*! \var int max_value_sequence
       *  \brief The max value. Compare to min_value_sequence
       *
       *
       */
      int max_value_sequence;

      /*! \var int gram_length_sequence
       *  \brief The length of each gram.
       */
      int gram_length_sequence;


public:

	/*! \fn inv_table()
	 *  \brief Default constructor of the inv_table.
	 *
     	 *  It sets is_store_in_gpu to false, _shifter 16,
         *  _size -1, table_index 0, total_num_of_table 1 and _dim_size 0.
	 */
	inv_table():is_stored_in_gpu(false),shift_bits_sequence(0),
                table_index(0),total_num_of_table(1),
                _build_status(not_builded), _shifter(16),_size(-1),_dim_size(0),
		        shift_bits_subsequence(0),
                min_value_sequence(0), max_value_sequence(0), gram_length_sequence(1)
	{
	}

    /*! \fn ~inv_table()
     *  \brief The Destructor of the inv_table. It will also clear the related gpu memory.
     */
	~inv_table();


    /*! \fn vector<vector<int> >* distinct_value()
     *  \brief return the distinct value vector for all inv_lists
     *
     *  \param dim The specific dim of the expected distinct values
     *
     */
    unordered_map<int, int>* get_distinct_map(int dim);

    /*! \fn void append_sequence(inv_list& inv)
     *  \brief append inv_list for sequence search
     *
     *  \param inv The inv_list to append.
     *
     */
    void append_sequence(inv_list& inv);

    /*! \fn static bool write(const char* filename, inv_table*& table)
     *  \brief static member function responsible for serialize inv_table objects to a binary file
     *
     *  \param filename The file to write.
     *  \param table The table to be serialized. It should be the first table, if there are more than
     *  one table.
     *
     *  \return True for successful operations, false for unsuccessful.
     */
    static bool
    write(const char* filename, inv_table*& table);

    /*! \fn static bool read(const char* filename, inv_table*& table)
     *  \brief static member function responsible for deserialize inv_table objects from a binary file
     *
     *  \param filename The file to read.
     *  \param table The table pointer. If the function finishes successfully, the 'table' would points
     *  to the table recorded by the binary file.
     *
     *  \return True for successful operations, false for unsuccessful.
     */
    static bool
    read(const char* filename, inv_table*& table);

    /*! \fn void set_table_index(int index)
     *  \brief Set the table_index to 'index'
     *  \param index The index value you wish to set.
     *
     *  Actually, users do not need to call this function.
     */
    void
    set_table_index(int index);

    /*! \fn void set_total_num_of_table(int num)
     *  \brief Set the total_num_of_table to 'num'
     *  \param num The total number of tables you wish to set.
     *
     *  Actually, users do not need to call this function.
     */
    void
    set_total_num_of_table(int num);

    /*! \fn int get_table_index()     *
     *  \brief return the index of this inv_table.
     *
     *  \return The index of this table.
     */
    int
    get_table_index();

    /*! \fn int get_total_num_of_table()
     *  \brief return the total_num_of_table.
     *
     *  \return The total number of tables in the table array for one dataset.
     */
    int
    get_total_num_of_table();


    /*! \fn bool cpy_data_to_gpu()
     *  \brief Copy vector _inv to gpu memory which is referenced by d_inv_p
     *
     *  \return True if transferring is successful.
     */
	bool
	cpy_data_to_gpu();

    /*! \fn void clear_gpu_mem()
     *  \brief clear the corresponding gpu memory referenced by d_inv_p
     */
	void
	clear_gpu_mem();

	/*! \fn void clear()
	 *  \brief Clear the inv_table
	 *
     *  The method removes all content in _inv_lists, _ck and _inv.
     *  It will release the corresponding gpu memory allocated as well.
	 *  It also sets the _size back to -1.
	 */
	void
	clear();

	/*! \fn bool empty()
	 *  \brief Check whether the inv_table is empty.
	 *
     *  If the _size is smaller than 0, the method returns true and returns false
	 *  in the other case.
	 *
     *  \return true if _size < 0
	 *         false if _size >= 0
	 */
	bool
	empty();

	/*! \fn int m_size()
	 *
     	 *  \return The number of dimensions.
	 */
	int
	m_size();

	/*! \fn int i_size()
	 *
     	 *  \return The number of instances(data points)
	 */
	int
	i_size();

	/*! \fn int shifter()
	 *  \return The shifter.
	 */
	int
	shifter();

	  /*! \fn unsigned int _shift_bits_subsequence()
    	   *  \return The shift bits for subsequence search. The way to combine
     	   *  rowID and offset of its element.
           */
    	unsigned int
    	_shift_bits_subsequence();

	/*! \fn void append(inv_list& inv)
	 *  \brief Append an inv_list to the inv_table.
	 *
	 *  \param inv The refrence to the inv_list which will be appended.
         *  The appended inv_list will be added
	 *  to _inv_lists. The first inv_list will set
	 *  the _size to correct number. If _size is not
	 *  equal to the size of inv_list or -1. This
	 *  method will simply return and do nothing.
	 */
	void
	append(inv_list& inv);

	/*! \fn void append(inv_list* inv)
	 *  \brief Append an inv_list to the inv_table.
	 *
	 *  \param inv: the refrence to the inv_list which will be appended.
     	 *
     	 *  The appended inv_list will be added
	 *  to _inv_lists. The first inv_list will set
	 *  the _size to correct number. If _size is not
	 *  equal to the size of inv_list or -1. This
	 *  method will simply return and do nothing.
	 */
	void
	append(inv_list* inv);

	/*! \fn status build_status()
	 *
         *  \return Building status of the inv_table.
	 */
	status
	build_status();

	/*! \fn vector<inv_list>* inv_lists()
	 *
         *  \return The pointer points to _inv_lists vector.
	 */
	vector<inv_list>*
	inv_lists();

	/*! \fn vector<int>* ck()
	 *
         *  \return The pointer points to _ck vector.
	 */
	vector<int>*
	ck();

	/*! \fn vector<int>* inv()
	 *
         *  \return The pointer points to _inv vector.
	 */
	vector<int>*
	inv();

	/*! \fn vector<int>* inv_index()
	 *
         *  \return The pointer points to _inv_index vector.
	 */
	vector<int>*
	inv_index();

	unordered_map<size_t, int>*
	inv_index_map();

        /*! \fn vector<int>* inv_pos()
	 *
         *  \return The pointer points to _inv_pos vector.
	 */
	vector<int>*
	inv_pos();


    /*! \fn int get_upperbound_of_list(int index)
     *
     *  \param index Specify index of the inverted list
     *
     *  \return The maximum value of the inverted list at 'index'
     */
    int
    get_upperbound_of_list(int index);

    /*! \fn int get_lowerbound_of_list(int index)
     *
     *  \param index Specify index of the inverted list
     *
     *  \return The minimum value of the inverted list at 'index'
     */
    int
    get_lowerbound_of_list(int index);




	/*! \fn void build(u64 max_length, bool use_load_balance)
	 *  \brief Build the inv_table.
	 *
         *  \param max_length The maximum length of one segment in posting list array
         *  \param use_load_balance The flag to determine whether to do load balance
         *
         *  This method will merge all inv_lists to
	 *  two vector _ck and _inv and set the
	 *  _build_status to builded. Any query should
	 *  only be done after the inv_table has been builded.
         *  If use_load_balance is true, the max_length will be used
         *  to divide posting list which is longer than max_length suggests.
         *  if use_load_balance is false, the max_length will be set
         *  to positive infinity before it is used
         *
         */
	void
	build(u64 max_length, bool use_load_balance);


    /*! \fn int get_posting_list_size(int attr_index, int value)
     *
     *  \param attr_index The attribute id
     *  \param value The value on that specific attribute
     *
     *  This function is called when constructing query.
     *  Users do not need to call this function.
     *
     *  \return The posting list length for an attribute-value pair
     */
    int
    get_posting_list_size(int attr_index, int value);

    /*! \fn bool list_contain(int attr_index, int value)
     *  \brief Test whether a value is possible for an specific attribute
     *
     *  \param attr_index Id of attribute
     *  \param value Value waiting to be tested
     *
     *  \return True if value is in the range allowed by the attribute
     */
    bool
    list_contain(int attr_index, int value);

    /*! \fn bool write_to_file(ofstream& ofs)
     *  \brief Write one table object to a binary file
     *
     *  \param ofs An ofstream object
     *
     *  This function is always called by static write function,
     *  which is a static member of inv_table class.
     *
     */
    bool
    write_to_file(ofstream& ofs);

    /*! \fn bool read_from_file(ifstream& ifs)
     *  \brief Read one table from a binary file
     *
     *  \param ifs An ifstream object
     *
     *  This function is always called by static read function,
     *  which is a static member of inv_table class.
     *
     */
    bool
    read_from_file(ifstream& ifs);


    /*! \fn void set_min_value_sequence(int min_value)
     *  \brief Used in sequence search. To set the min_value for all sequences' element.
     *
     *  \param min_value The value to be set as the minimum value.
     */
    void
    set_min_value_sequence(int min_value);

    /*! \fn int get_min_value_sequence()
     *  \brief Get the min value for sequences' elements in this inv_table.
     *
     *  \return The min value.
     */
    int
    get_min_value_sequence();

    /*! \fn void set_max_value_sequence(int max_value)
     *  \brief Set the max value for all sequence. Compare to set_min_value_sequence()
     *
     *  \param max_value The max value to be set.
     */
    void
    set_max_value_sequence(int max_value);

    /*! \fn int get_max_value_sequence()
     *  \brief Get the max value.
     *
     *  \return The max value.
     */
    int
    get_max_value_sequence();

    /*! \fn void set_gram_length_sequence(int gram_length)
     *  \brief Set length of each gram.
     *
     *  \param gram_length The gram length to be set.
     *
     */
    void
    set_gram_length_sequence(int gram_length);

    /*! \fn int get_gram_length_sequence()
     *
     *  \brief Get the gram length.
     *
     *  \return The gram length used in this inv_table.
     *
     */
    int
    get_gram_length_sequence();






};
}


#endif
