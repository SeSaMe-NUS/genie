
#ifndef PERF_LOGGER_H_
#define PERF_LOGGER_H_

#include <assert.h>
#include <fstream>
#include <string>

namespace GPUGenie
{
    
class PerfLogger{
public:
    static PerfLogger& get()
    {
        static PerfLogger pl;
        return pl;
    }

    bool setOutputFileStream(std::ofstream &ofs)
    {
        m_OFS = &ofs;
        return ofs.good();
    }

    std::ofstream& ofs(){
        return *m_OFS;
    }

    ~PerfLogger(){};

private:
    PerfLogger() : m_OFS(&m_nullOFS) {};
    PerfLogger(PerfLogger const&);
    PerfLogger& operator=(PerfLogger const&);

    std::ofstream m_nullOFS;
    std::ofstream *m_OFS;
};

}
#endif
