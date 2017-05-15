
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
        assert(!m_ofs);
        m_ofs = &ofs;
        return ofs.good();
    }

    std::ofstream& ofs(){
        assert(m_ofs);
        return *m_ofs;
    }

    ~PerfLogger(){};

private:
    PerfLogger(){};
    PerfLogger(PerfLogger const&);
    PerfLogger& operator=(PerfLogger const&);

    std::ofstream *m_ofs;
};

}
#endif
