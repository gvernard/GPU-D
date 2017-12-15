#ifndef _OOC_DEVICE_H
#define _OOC_DEVICE_H

#include "string.hpp"

namespace mycuda {

	class Device{

		int m_idx;
		cudaDeviceProp m_props;
	
	public:
		Device(int idx) : m_idx(idx){
			cudaError cuda_ret;
			
			cuda_ret = cudaGetDeviceProperties(&m_props, m_idx);
			if( cudaSuccess != cuda_ret ) {
			  throw string("Failed to aquire device properties (device ")+toString(m_idx)+"): "+cudaGetErrorString(cuda_ret);
			}
		}

		~Device() {}

		void initialise() const{
			cudaError cuda_ret;
			
			cuda_ret = cudaSetDevice(m_idx);
			if( cudaSuccess != cuda_ret ) {
			  throw string("Failed to set device to ")+toString(m_idx)+": "+cudaGetErrorString(cuda_ret);
			}
		}

		int    getIndex()             const { return m_idx; }
		string getName()              const { return m_props.name; }
		size_t getTotalGlobalMem()    const { return m_props.totalGlobalMem; }
		size_t getProcessorCount()    const { return m_props.multiProcessorCount; }
		size_t getClockRate()         const { return m_props.clockRate*1000; }
		size_t getTotalConstMem()     const { return m_props.totalConstMem; }
		size_t getSharedMemPerBlock() const { return m_props.sharedMemPerBlock; }
		size_t getMaxFLOPS()          const { return this->getProcessorCount()*8*this->getClockRate(); }

	};


} //namespace mycuda

#endif //_OOC_DEVICE_H
