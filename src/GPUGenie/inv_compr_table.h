/*! \file inv_compr_table.h
 *  \brief define class inv_compre_table
 *
 *  This file contains the declaration for inv_compr_table class
 */


#ifndef GPUGenie_inv_compr_table_h
#define GPUGenie_inv_compr_table_h

#include "inv_table.h"

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

class inv_compr_table : public inv_table
{
protected:

    bool _is_compressed;


public:

    /*! \fn inv_compr_table()
     *  \brief Default constructor of the inv_compr_table.
     */
    inv_compr_table(): inv_table(),_is_compressed(false){}

    /*! \fn ~inv_table()
     *  \brief The Destructor of the inv_table. It will also clear the related gpu memory.
     */
    ~inv_compr_table();


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

};
}


#endif
