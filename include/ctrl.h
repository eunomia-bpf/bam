
#ifndef __BENCHMARK_CTRL_H__
#define __BENCHMARK_CTRL_H__

// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "buffer.h"
#include "nvm_types.h"
#include "nvm_ctrl.h"
#include "nvm_aq.h"
#include "nvm_admin.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <algorithm>
#include <simt/atomic>

#include "queue.h"


#define MAX_QUEUES 1024


struct Controller
{
    simt::atomic<uint64_t, simt::thread_scope_device> access_counter;
    nvm_ctrl_t*             ctrl;
    nvm_aq_ref              aq_ref;
    DmaPtr                  aq_mem;
    struct nvm_ctrl_info    info;
    struct nvm_ns_info      ns;
    uint16_t                n_sqs;
    uint16_t                n_cqs;
    uint16_t                n_qps;
    uint32_t                deviceId;
    QueuePair**             h_qps;
    QueuePair*              d_qps;

    simt::atomic<uint64_t, simt::thread_scope_device> queue_counter;

    uint32_t page_size;
    uint32_t blk_size;
    uint32_t blk_size_log;


    void* d_ctrl_ptr;
    BufferPtr d_ctrl_buff;
#ifdef __DIS_CLUSTER__
    Controller(uint64_t controllerId, uint32_t nvmNamespace, uint32_t adapter, uint32_t segmentId);
#endif

    Controller(const char* path, uint32_t nvmNamespace, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues);

    void reserveQueues();

    void reserveQueues(uint16_t numSubmissionQueues);

    void reserveQueues(uint16_t numSubmissionQueues, uint16_t numCompletionQueues);

    void print_reset_stats(void);

    ~Controller();
};



using error = std::runtime_error;
using std::string;


inline void Controller::print_reset_stats(void) {
    cuda_err_chk(cudaMemcpy(&access_counter, d_ctrl_ptr, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaMemcpyDeviceToHost));
    std::cout << "------------------------------------" << std::endl;
    std::cout << std::dec << "#SSDAccesses:\t" << access_counter << std::endl;

    cuda_err_chk(cudaMemset(d_ctrl_ptr, 0, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>)));
}

static void initializeController(struct Controller& ctrl, uint32_t ns_id)
{
    // Create admin queue reference
    int status = nvm_aq_create(&ctrl.aq_ref, ctrl.ctrl, ctrl.aq_mem.get());
    if (!nvm_ok(status))
    {
        throw error(string("Failed to reset controller: ") + nvm_strerror(status));
    }

    // Identify controller
    status = nvm_admin_ctrl_info(ctrl.aq_ref, &ctrl.info, NVM_DMA_OFFSET(ctrl.aq_mem, 2), ctrl.aq_mem->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    // Identify namespace
    status = nvm_admin_ns_info(ctrl.aq_ref, &ctrl.ns, ns_id, NVM_DMA_OFFSET(ctrl.aq_mem, 2), ctrl.aq_mem->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    // Get number of queues
    status = nvm_admin_get_num_queues(ctrl.aq_ref, &ctrl.n_cqs, &ctrl.n_sqs);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }
}



#ifdef __DIS_CLUSTER__
Controller::Controller(uint64_t ctrl_id, uint32_t ns_id, uint32_t)
    : ctrl(nullptr)
    , aq_ref(nullptr)
{
    // Get controller reference
    int status = nvm_dis_ctrl_init(&ctrl, ctrl_id);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to get controller reference: ") + nvm_strerror(status));
    }

    // Create admin queue memory
    aq_mem = createDma(ctrl, ctrl->page_size * 3, 0, 0);

    initializeController(*this, ns_id);
}
#endif



inline Controller::Controller(const char* path, uint32_t ns_id, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues)
    : ctrl(nullptr)
    , aq_ref(nullptr)
    , deviceId(cudaDevice)
{
    int fd = open(path, O_RDWR);
    if (fd < 0)
    {
        throw error(string("Failed to open descriptor: ") + strerror(errno));
    }

    // Get controller reference
    int status = nvm_ctrl_init(&ctrl, fd);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to get controller reference: ") + nvm_strerror(status));
    }

    // Create admin queue memory
    aq_mem = createDma(ctrl, ctrl->page_size * 3);

    initializeController(*this, ns_id);
    
    // Set CUDA device before registering memory
    cudaError_t setdev_err = cudaSetDevice(cudaDevice);
    if (setdev_err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(setdev_err));
    }
    
    // Initialize CUDA runtime
    cudaFree(0);
    
    // Debug output
    printf("DEBUG: Attempting cudaHostRegister with:\n");
    printf("  CUDA Device: %d\n", cudaDevice);
    printf("  mm_ptr: %p\n", ctrl->mm_ptr);
    printf("  mm_size: %lu\n", ctrl->mm_size);
    printf("  NVM_CTRL_MEM_MINSIZE: %lu (0x%lx)\n", (size_t)NVM_CTRL_MEM_MINSIZE, (size_t)NVM_CTRL_MEM_MINSIZE);
    printf("  mm_ptr alignment: %lu\n", (uintptr_t)ctrl->mm_ptr % 4096);
    
    // Check if pointer is valid
    if (ctrl->mm_ptr == nullptr)
    {
        throw error(string("Controller memory pointer is NULL"));
    }
    
    // Check if memory is already registered
    cudaPointerAttributes attrs;
    cudaError_t check_err = cudaPointerGetAttributes(&attrs, (void*)ctrl->mm_ptr);
    if (check_err == cudaSuccess && attrs.type != cudaMemoryTypeUnregistered)
    {
        printf("WARNING: Memory already registered with type %d\n", attrs.type);
        // Memory is already registered, skip registration
    }
    else
    {
        // Clear the error from cudaPointerGetAttributes if it failed
        cudaGetLastError();
        
        // Try registering with IoMemory flag first
        cudaError_t err = cudaHostRegister((void*) ctrl->mm_ptr, NVM_CTRL_MEM_MINSIZE, cudaHostRegisterIoMemory);
        if (err != cudaSuccess)
        {
            printf("ERROR: cudaHostRegister with IoMemory failed with error code %d: %s\n", err, cudaGetErrorString(err));
            
            // Try with Mapped flag instead
            printf("Trying with cudaHostRegisterMapped flag...\n");
            err = cudaHostRegister((void*) ctrl->mm_ptr, NVM_CTRL_MEM_MINSIZE, cudaHostRegisterMapped);
            if (err != cudaSuccess)
            {
                printf("ERROR: cudaHostRegisterMapped failed with error code %d: %s\n", err, cudaGetErrorString(err));
                
                // Try with Default flag
                printf("Trying with cudaHostRegisterDefault flag...\n");
                err = cudaHostRegister((void*) ctrl->mm_ptr, NVM_CTRL_MEM_MINSIZE, cudaHostRegisterDefault);
                if (err != cudaSuccess)
                {
                    printf("ERROR: cudaHostRegisterDefault failed with error code %d: %s\n", err, cudaGetErrorString(err));
                    
                    // Last resort: try to skip registration entirely
                    printf("WARNING: Unable to register memory with CUDA. Proceeding without registration.\n");
                    printf("This may impact performance or cause issues with GPU access.\n");
                    // Don't throw error, just continue
                }
                else
                {
                    printf("Success with cudaHostRegisterDefault!\n");
                }
            }
            else
            {
                printf("Success with cudaHostRegisterMapped!\n");
            }
        }
        else
        {
            printf("Success with cudaHostRegisterIoMemory!\n");
        }
    }
    queue_counter = 0;
    page_size = ctrl->page_size;
    blk_size = this->ns.lba_data_size;
    blk_size_log = std::log2(blk_size);
    reserveQueues(MAX_QUEUES,MAX_QUEUES);
    n_qps = std::min(n_sqs, n_cqs);
    n_qps = std::min(n_qps, (uint16_t)numQueues);
    // printf("SQs: %d\tCQs: %d\tn_qps: %d\n", n_sqs, n_cqs, n_qps);
    h_qps = (QueuePair**) malloc(sizeof(QueuePair)*n_qps);
    cuda_err_chk(cudaMalloc((void**)&d_qps, sizeof(QueuePair)*n_qps));
    for (size_t i = 0; i < n_qps; i++) {
        //printf("started creating qp\n");
        h_qps[i] = new QueuePair(ctrl, cudaDevice, ns, info, aq_ref, i+1, queueDepth);
        //printf("finished creating qp\n");
        cuda_err_chk(cudaMemcpy(d_qps+i, h_qps[i], sizeof(QueuePair), cudaMemcpyHostToDevice));
    }
    //printf("finished creating all qps\n");


    close(fd);

    d_ctrl_buff = createBuffer(sizeof(Controller), cudaDevice);
    d_ctrl_ptr = d_ctrl_buff.get();
    cuda_err_chk(cudaMemcpy(d_ctrl_ptr, this, sizeof(Controller), cudaMemcpyHostToDevice));
}



inline Controller::~Controller()
{
    cudaFree(d_qps);
    for (size_t i = 0; i < n_qps; i++) {
        delete h_qps[i];
    }
    free(h_qps);
    nvm_aq_destroy(aq_ref);
    nvm_ctrl_free(ctrl);

}



inline void Controller::reserveQueues()
{
    reserveQueues(n_sqs, n_cqs);
}



inline void Controller::reserveQueues(uint16_t numSubmissionQueues)
{
    reserveQueues(numSubmissionQueues, n_cqs);
}



inline void Controller::reserveQueues(uint16_t numSubs, uint16_t numCpls)
{
    int status = nvm_admin_request_num_queues(aq_ref, &numSubs, &numCpls);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to reserve queues: ") + nvm_strerror(status));
    }

    n_sqs = numSubs;
    n_cqs = numCpls;

}



#endif
