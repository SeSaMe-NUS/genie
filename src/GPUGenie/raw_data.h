#ifndef GPUGenie_raw_data_h
#define GPUGenie_raw_data_h

#include <vector>
#include <string>

using namespace std;

namespace GPUGenie
{
  struct raw_data
  {
    /**
     * @brief The meta information.
     * @details The meta information or the attributes' names.
     */
    vector<string> _meta;

    /**
     * @brief Instances.
     * @details Instances or rows.
     */
    vector<vector<string> > _instance;

    /**
     * @brief Columns.
     * @details The transposition matrix of the _instance's matrix.
     */
    vector<vector<string> > _transpose;

    /**
     * @brief Clear the content in raw_data.
     * @details This method removes all contents in _meta, _instance
     *          and _transpose.
     */
    void
    clear();

    /**
     * @brief The size of meta information.
     * @details The size of meta information. It directly returns the
     *          _meta vector's size.
     * @return The size of meta information.
     */
    int
    m_size();

    /**
     * @brief The size of instances.
     * @details The size of instances. It directly returns the
     *          _instance vector's size.
     * @return The size of instances.
     */
    int
    i_size();

    /**
     * @brief The meta information at specific index.
     * @details A pointer points to the meta information at specific index.
     *          Return NULL if the information is not available.
     * 
     * @param attr_index The specific index.
     * @return A pointer points to the meta information or a NULL pointer if
     *         the information is not available.
     */
    string*
    meta(int attr_index);

    /**
     * @brief The index of the specific meta information.
     * @details Return the index of the given meta information. Return
     *          -1 if the meta information is not found in the raw_data.
     * 
     * @param attr The given meta information.
     * @return The index of the given meta information or -1 if it cannot
     *         be found.
     */
    int
    meta(string attr);

    /**
     * @brief The instance which has the given row index.
     * @details The instance at the given index. Return a NULL pointer if
     *          the index is out of range.
     * 
     * @param index The row index.
     * @return A pointer points to the specific row.
     */
    vector<string>*
    row(int index);

    /**
     * @brief The column of given attribute.
     * @details The column of given attribute. The method will check the
     *          _transpose vector to get the column information. The method
     *          will first check whether _transpose vector is empty. It will first
     *          build the _transpose vector if it is empty or simply return the column's
     *          pointer if the _transpose vector has been already builded.
     * 
     * @param attr The attribute string.
     * @return A pointer points to the column. Return NULL if the attribute does not exist.
     */
    vector<string>*
    col(string attr);

    /**
     * @brief The column of given attribute index.
     * @details The column of given attribute index. The method will check the
     *          _transpose vector to get the column information. The method
     *          will first check whether _transpose vector is empty. It will first
     *          build the _transpose vector if it is empty or simply return the column's
     *          pointer if the _transpose vector has been already builded.
     * 
     * @param attr_index The attribute index.
     * @return A pointer points to the column. Return NULL if the attribute does not exist.
     */
    vector<string>*
    col(int attr_index);
  };
}

#endif
