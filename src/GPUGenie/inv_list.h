/*! \file inv_list.h
 *  \brief Declaration of inv_list class
 */

#ifndef GPUGenie_inv_list_h
#define GPUGenie_inv_list_h

#include <vector>
#include <string>

using namespace std;

namespace GPUGenie
{

/*! \class inv_list
 *  \brief This class manages one inverted list
 *
 *  This class contains information about one attribute and value range of the attribute.
 *  It also stores id of data points that contain a specific value on that attribute.
 */
class inv_list
{
private:
	/*! \var int _size
	 *  \brief The number of instances in the original vector.
	 */
	int _size;

	/*! \var std::pair<int, int> _bound
	 *  \brief The min max values.
	 */
	std::pair<int, int> _bound;

	/*! \var vector<vector<int> > _inv
	 *  \brief The inverted list vector.
     *
     *  This vector contains all the data points that have a value on the attribute. The inv_table contains many inv_list objects
     *  Thus, inv_list is the basis of inv_table.
	 */
	vector<vector<int> > _inv;

public:
	/*! \fn inv_list()
	 *  \brief Create an empty inv_list.
	 */
	inv_list() :
			_size(0)
	{
	}

	/*! \fn inv_list(vector<int>& vin)
	 *  \brief Create an inv_list from an int vector.
	 *
	 *  \param vin The vector which will be inverted.
	 */
	inv_list(vector<int>& vin);

	/*! \fn inv_list(vector<inv>* vin)
	 *  \brief Create an inv_list from an int vector.
	 *
	 *  \param vin The vector which will be inverted.
	 */
	inv_list(vector<int>* vin);

    /*! \fn invert_bijectMap(vector<vector<int> > & vin)
     *  \brief invert a special kind of data,which is of one dimension
     *
     *  \vin Data to be inverted
     *
     *  This function will be used to invert data points like short text data,
     *  which has only one dimension but many individual words on that attibute.
     *  Data is from a csv file.
     */
	void
	invert_bijectMap(vector<vector<int> > & vin);

	/*! \fn void invert_bijectMap(int *data, unsigned int item_num, unsigned int *index, unsigned int row_num)
	 *  \brief Handle input from binary file
	 *
     *  \param data The data array
     *  \param Length of data array
     *  \param index Stores the starting position of each data point in data array
     *  \param row_num Length of index array
     *
     *  This function does the same thing as invert_bijectMap(vector<vector<int> > & vin), but the input is from a
     *  binary form.
     */
	void
	invert_bijectMap(int *data, unsigned int item_num, unsigned int *index, unsigned int row_num);

	/*! \fn inv_list(vector<string>& vin)
	 *  \brief Create an inv_list from a string vector.
	 *
	 *  \param vin The vector which will be inverted.
     *
     *  The default converter atoi will be invoked to convert the string value to int value.
	 */
	inv_list(vector<string>& vin);

	/*! \fn inv_list(vector<string>* vin)
	 *  \brief Create an inv_list from a string vector.
     *
	 *   \param vin The vector which will be inverted.
     *
     *   The default converter atoi will be invoked to convert the string value to int value.
	 */
	inv_list(vector<string>* vin);

	/*! \fn int min()
	 *  \brief Return the min value of the inverted vector.
	 *
     *  \return The min value of the inverted vector.
	 */
	int
	min();

	/*! \fn int max()
	 *  \brief Return the max value of the inverted vector.
	 *
     *  \return The max value of the inverted vector.
	 */
	int
	max();

	/*! \fn int size()
	 *  \brief Return the number of instances.
	 *
     *  \return The number of instances.
	 */
	int
	size();

	/*! \fn void invert(vector<int>& vin)
	 *  \brief Create an inverted list from an int vector.
	 *
	 *  \param vin The vector which will be inverted.
	 */
	void
	invert(vector<int>& vin);

	/*! \fn void invert(vector<int>& vin)
	 *  \brief Create an inverted list from an int vector.
	 *
	 *  \param vin The vector which will be inverted.
	 */
	void
	invert(vector<int>* vin);

	/*! \fn void invert(vector<string>& vin)
	 *  \brief Create an inverted list from a string vector.
	 *
	 *   \param vin The vector which will be inverted.
     *
     *   The default converter atoi will be invoked to convert the string value to int value.
	 */
	void
	invert(vector<string>& vin);

	/*! \fn void invert(vector<string>* vin)
	 *  \brief Create an inverted list from a string vector.
	 *
	 *  \param vin The vector which will be inverted.
     *
     *  The default converter atoi will be invoked to convert the string value to int value.
	 */
	void
	invert(vector<string>* vin);

	/*! \fn void invert(vector<string>& vin, int (*stoi)(string&))
	 *  \brief Create an inverted list from a string vector.
	 *
     *
	 *   \param vin The vector which will be inverted.
	 *   \param stoi The converter function pointer.
     *
     *   The void pointer will be passed to the converter function.
	 *   For example, if the converter converts the string to int based on
	 *   the min max value, the void pointer shall point to the structure which contains min max
	 *   then in the converter function stoi, downcast the void pointer to the min max structure.
	 */
	void
	invert(vector<string>& vin, int (*stoi)(string&));

	/*! \fn void invert(vector<string>* vin, int (*stoi)(string&))
	 *  \brief Create an inverted list from a string vector.
	 *
	 *  \param vin The vector which will be inverted.
	 *  \param stoi The converter function pointer.
     *
     *  The void pointer will be passed to the converter function.
	 *  For example, if the converter converts the string to int based on
	 *  the min max value, the void pointer shall point to the structure which contains min max
	 *  then in the converter function stoi, downcast the void pointer to the min max structure.
	 *
	 */
	void
	invert(vector<string>* vin, int (*stoi)(string&));

	/*! \fn bool contains(int value)
	 *  \brief Check whether the vaule is in the inv_list.
	 *
	 *  \param  value The given value.
	 *
     *  \return True only if the vaule is in the inv_list.
	 */
	bool
	contains(int value);

	/*! \fn vector<int>* index(int value)
	 *  \brief The indexes of the value.
	 *
	 *  \param value The given value.
     *
     *  The value's indexes in the original vector. Return NULL if the given
	 *  value does not appear in the original vector.
	 *
	 *
     *  \return Pointer points to the indexes vector. NULL if the value does not appear
	 *         in the original vector.
	 */
	vector<int>*
	index(int value);

    /*! \fn int value_range()
     *  \brief Return the number of different values in the attribute
     *
     * \return Number of different values in the attribute
     */
    int
    value_range();

};
}

#endif
