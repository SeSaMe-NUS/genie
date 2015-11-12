#ifndef GaLG_match_h
#define GaLG_match_h

#include "../GaLG.h"
#include <stdint.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

using namespace std;
using namespace thrust;

const char VERSION[] = "0.2.5";
typedef unsigned char u8;
typedef uint32_t u32;
typedef unsigned long long u64;
class MemException: public std::exception {
private:
    std::string message_;
public:
    explicit MemException(const std::string& message){message_ = std::string(message);}
    explicit MemException(const char * message){message_ = std::string(message);}
    explicit MemException(char * message){message_ = std::string(message);}
    virtual const char* what() const throw() {
        return message_.c_str();
    }
    virtual ~MemException() throw(){;}
};
#define cudaCheckErrors( err ) __cudaSafeCall( err, __FILE__, __LINE__ )


inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{

    if ( cudaSuccess != err )
    {
    	char errstr[1000];
        sprintf( errstr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        printf("cudaSafeCall failed in match function!\n");
        throw MemException(errstr);
    }


    return;
}


inline cudaError checkAndMalloc(void ** to, u64 bytes)
{
	size_t f, t;
	cudaMemGetInfo(&f, &t);
	if(f <= bytes)
	{
		throw MemException("Insufficient GPU memory...");
	}
	return cudaMalloc(to, bytes);
}


u64 getTime();
double getInterval(u64 t1, u64 t2);
namespace GaLG
{

  /**
   * @brief Search the inv_table and save the match
   *        result into d_count and d_aggregation.
   * @details Search the inv_table and save the match
   *          result into d_count and d_aggregation.
   * 
   * @param table The inv_table which will be searched.
   * @param queries The quries.
   * @param d_data The output data consisting of count, aggregation
   *               and the index of the data in big table.
   * @param hash_table_size The hash table size.
   *
   * @throw inv_table::not_builded_exception if the table has not been builded.
   * @throw inv_table::not_matched_exception if the query is not querying the given table.
   */
	void
	match(inv_table& table,
		  vector<query>& queries,
		  device_vector<data_t>& d_data,
		  int hash_table_size,
		  int max_load,
		  int bitmap_bits,
		  int num_of_hot_dims,
		  int hot_dim_threshold,
		  device_vector<u32>& d_noiih);
  void
  match(inv_table& table,
        vector<query>& queries,
        device_vector<data_t>& d_data,
        device_vector<u32>& d_bitmap,
        int hash_table_size,
        int max_load,
        int bitmap_bits,
        int num_of_hot_dims,
        int hot_dim_threshold,
        device_vector<u32>& d_noiih);
  void
  build_queries(vector<query>& queries, inv_table& table, vector<query::dim>& dims, int max_load);

}

#endif
