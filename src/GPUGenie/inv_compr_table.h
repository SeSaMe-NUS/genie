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


class inv_compr_table : public inv_table
{
protected:

    bool m_isCompressed;

    std::vector<uint32_t> m_comprInv;

    std::vector<int> m_comprInvPos;

    std::string m_compression;

    int *m_d_compr_inv_p;

    size_t m_uncompressedInvListsMaxLength;


public:

    /*! \fn inv_compr_table()
     *  \brief Default constructor of the inv_compr_table.
     */
    inv_compr_table(): inv_table(),
            m_isCompressed(false),
            m_d_compr_inv_p(NULL),
            m_uncompressedInvListsMaxLength(0){}

    /*! \fn ~inv_table()
     *  \brief The Destructor of the inv_table. It will also clear the related gpu memory.
     */
    virtual ~inv_compr_table();


    const std::string& getCompression() const;

    void setCompression(const std::string &compression);


    /*
     * Return the equivalent of GPUGenie_Config::posting_list_max_length, indicating what is the maximum length of
     * a single uncompressed posting lists.
     * TODO: make this actually return the maximum length of an existing uncompressed inverted list
     */
    size_t getUncompressedPostingListMaxLength() const;

    void setUncompressedPostingListMaxLength(size_t length);

    /* 
     * Returns compressed version of _inv (posting lists array)
     */
    std::vector<int>* compressedInv();

    /* 
     * Returns compressed version of _inv_pos (starting positions of inverted lists in posting lists array)
     */
    std::vector<int>* compressedInvPos();

    /* 
     * Returns _inv (the same as inv() function), as this index is on CPU and maps the domain of keys into the starting
     * positions of inverted lists -- i.e. compressedInvPod()
     */
    std::vector<int>* compressedInvIndex();
    
    /* 
     * Returns _ck
     */
    std::vector<int>* compressedCK();


    int* deviceCompressedInv() const;



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
    virtual void build(u64 max_length, bool use_load_balance);



    /*! \fn bool cpy_data_to_gpu()
     *  \brief Copy vector _inv to gpu memory which is referenced by d_inv_p
     *
     *  \return True if transferring is successful.
     */
    bool cpy_data_to_gpu();

    /*! \fn void clear_gpu_mem()
     *  \brief clear the corresponding gpu memory referenced by d_inv_p
     */
    void clear_gpu_mem();


    void clear();

};
}


#endif
