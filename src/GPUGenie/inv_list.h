#ifndef GPUGenie_inv_list_h
#define GPUGenie_inv_list_h

#include <vector>
#include <string>

using namespace std;

namespace GPUGenie
{
  class inv_list
  {
  private:
    /**
     * @brief The number of instances.
     * @details The number of instances or values in the original vector.
     */
    int _size;

    /**
     * @brief The min max values.
     * @details The min max values in the original vector.
     */
    pair<int, int> _bound;

    /**
     * @brief The inverted list vector.
     * @details The inverted list vector which maps a value to an index.
     */
    vector<vector<int> > _inv;

  public:
    /**
     * @brief Create an empty inv_list.
     * @details Create an empty inv_list.
     */
    inv_list() :
        _size(0)
    {
    }

    /**
     * @brief Create an inv_list from an int vector.
     * @details Create an inv_list from an int vector.
     * 
     * @param vin The vector which will be inverted.
     */
    inv_list(vector<int>& vin);

    /**
     * @brief Create an inv_list from an int vector.
     * @details Create an inv_list from an int vector.
     * 
     * @param vin The vector which will be inverted.
     */
    inv_list(vector<int>* vin);

    void
    invert_tweets(vector<vector<int> > & vin);

    /**
     * @brief Create an inv_list from a string vector.
     * @details Create an inv_list from a string vector.
     *          The default converter atoi will be invoked to 
     *          convert the string value to int value.
     * 
     * @param vin The vector which will be inverted.
     */
    inv_list(vector<string>& vin);

    /**
     * @brief Create an inv_list from a string vector.
     * @details Create an inv_list from a string vector.
     *          The default converter atoi will be invoked to 
     *          convert the string value to int value.
     * 
     * @param vin The vector which will be inverted.
     */
    inv_list(vector<string>* vin);

    /**
     * @brief The min value of the inverted vector.
     * @details The min value of the inverted vector.
     * @return The min value of the inverted vector.
     */
    int
    min();

    /**
     * @brief The max value of the inverted vector.
     * @details The max value of the inverted vector.
     * @return The max value of the inverted vector.
     */
    int
    max();

    /**
     * @brief The number of instances.
     * @details The number of instances or values in the original vector.
     * @return The number of instances.
     */
    int
    size();

    /**
     * @brief Create an inverted list from an int vector.
     * @details Create an inverted list from an int vector.
     * 
     * @param vin The vector which will be inverted.
     */
    void
    invert(vector<int>& vin);

    /**
     * @brief Create an inverted list from an int vector.
     * @details Create an inverted list from an int vector.
     * 
     * @param vin The vector which will be inverted.
     */
    void
    invert(vector<int>* vin);

    /**
     * @brief Create an inverted list from a string vector.
     * @details Create an inverted list from a string vector.
     *          The default converter atoi will be invoked to 
     *          convert the string value to int value.
     * 
     * @param vin The vector which will be inverted.
     */
    void
    invert(vector<string>&);

    /**
     * @brief Create an inverted list from a string vector.
     * @details Create an inverted list from a string vector.
     *          The default converter atoi will be invoked to 
     *          convert the string value to int value.
     * 
     * @param vin The vector which will be inverted.
     */
    void
    invert(vector<string>*);

    /**
     * @brief Create an inverted list from a string vector.
     * @details Create an inverted list from a string vector and a converter
     *          stoi. The void pointer will be passed to the converter function.
     *          For example, if the converter converts the string to int based on
     *          the min max value, the void pointer shall point to the structure which contains min max
     *          then in the converter function stoi, downcast the void pointer to the min max structure.
     * 
     * @param vin The vector which will be inverted.
     * @param stoi The converter function pointer.
     * @param d Anyother things that will be passed to the stoi function.
     */
    void
    invert(vector<string>& vin, int
    (*stoi)(string&, void*), void* d);

    /**
     * @brief Create an inverted list from a string vector.
     * @details Create an inverted list from a string vector and a converter
     *          stoi. The void pointer will be passed to the converter function.
     *          For example, if the converter converts the string to int based on
     *          the min max value, the void pointer shall point to the structure which contains min max
     *          then in the converter function stoi, downcast the void pointer to the min max structure.
     * 
     * @param vin The vector which will be inverted.
     * @param stoi The converter function pointer.
     * @param d Anyother things that will be passed to the stoi function.
     */
    void
    invert(vector<string>* vin, int
    (*stoi)(string&, void*), void* d);

    /**
     * @brief Check whether the vaule is in the inv_list.
     * @details Check whether the vaule is in the inv_list.
     * 
     * @param  value The given value.
     * @return Whether the vaule is in the inv_list.
     */
    bool
    contains(int value);

    /**
     * @brief The indexes of the value.
     * @details The value's indexes in the original vector. Return NULL if the given
     *          value does not appear in the original vector.
     * 
     * @param value The given value.
     * @return Pointer points to the indexes vector. NULL if the value does not appear
     *         in the original vector.
     */
    vector<int>*
    index(int value);

  };
}

#endif
