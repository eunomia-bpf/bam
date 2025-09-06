#ifndef NVIDIA_WRAPPER_H
#define NVIDIA_WRAPPER_H

#ifdef _CUDA

// Include the original header but don't link to symbols
#include <nv-p2p.h>

// Function pointer declarations
extern int (*nv_p2p_get_pages)(uint64_t, uint32_t, uint64_t, uint64_t, struct nvidia_p2p_page_table **, void (*)(void*), void*);
extern int (*nv_p2p_put_pages)(uint64_t, uint32_t, uint64_t, struct nvidia_p2p_page_table *);
extern int (*nv_p2p_free_page_table)(struct nvidia_p2p_page_table *);
extern int (*nv_p2p_dma_map_pages)(struct pci_dev *, struct nvidia_p2p_page_table *, struct nvidia_p2p_dma_mapping **);
extern int (*nv_p2p_dma_unmap_pages)(struct pci_dev *, struct nvidia_p2p_page_table *, struct nvidia_p2p_dma_mapping *);

// Wrapper macros
#define nvidia_p2p_get_pages nv_p2p_get_pages
#define nvidia_p2p_put_pages nv_p2p_put_pages
#define nvidia_p2p_free_page_table nv_p2p_free_page_table
#define nvidia_p2p_dma_map_pages nv_p2p_dma_map_pages
#define nvidia_p2p_dma_unmap_pages nv_p2p_dma_unmap_pages

int init_nvidia_symbols(void);
void cleanup_nvidia_symbols(void);

#endif /* _CUDA */
#endif /* NVIDIA_WRAPPER_H */
