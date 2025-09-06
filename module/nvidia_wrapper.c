#include <linux/module.h>
#include <linux/kallsyms.h>
#include <linux/kprobes.h>
#include "nvidia_wrapper.h"

#ifdef _CUDA

// Function pointers
int (*nv_p2p_get_pages)(uint64_t, uint32_t, uint64_t, uint64_t, struct nvidia_p2p_page_table **, void (*)(void*), void*) = NULL;
int (*nv_p2p_put_pages)(uint64_t, uint32_t, uint64_t, struct nvidia_p2p_page_table *) = NULL;
int (*nv_p2p_free_page_table)(struct nvidia_p2p_page_table *) = NULL;
int (*nv_p2p_dma_map_pages)(struct pci_dev *, struct nvidia_p2p_page_table *, struct nvidia_p2p_dma_mapping **) = NULL;
int (*nv_p2p_dma_unmap_pages)(struct pci_dev *, struct nvidia_p2p_page_table *, struct nvidia_p2p_dma_mapping *) = NULL;

// Use kprobes to find kallsyms_lookup_name
typedef unsigned long (*kallsyms_lookup_name_t)(const char *name);
static kallsyms_lookup_name_t kallsyms_lookup_name_ptr = NULL;

static struct kprobe kp = {
    .symbol_name = "kallsyms_lookup_name"
};

static int find_kallsyms_lookup_name(void)
{
    int ret;
    
    memset(&kp, 0, sizeof(kp));
    kp.symbol_name = "kallsyms_lookup_name";
    
    ret = register_kprobe(&kp);
    if (ret < 0) {
        printk(KERN_WARNING "Failed to register kprobe: %d\n", ret);
        return ret;
    }
    
    kallsyms_lookup_name_ptr = (kallsyms_lookup_name_t)kp.addr;
    unregister_kprobe(&kp);
    
    if (!kallsyms_lookup_name_ptr) {
        printk(KERN_WARNING "Failed to find kallsyms_lookup_name\n");
        return -ENOENT;
    }
    
    return 0;
}

int init_nvidia_symbols(void)
{
    int ret;
    
    // First, find kallsyms_lookup_name using kprobes
    ret = find_kallsyms_lookup_name();
    if (ret != 0) {
        printk(KERN_WARNING "Cannot find kallsyms_lookup_name, trying direct symbol resolution\n");
        // Try to use symbol_get instead
        nv_p2p_get_pages = (void*)symbol_get(nvidia_p2p_get_pages);
        nv_p2p_put_pages = (void*)symbol_get(nvidia_p2p_put_pages);
        nv_p2p_free_page_table = (void*)symbol_get(nvidia_p2p_free_page_table);
        nv_p2p_dma_map_pages = (void*)symbol_get(nvidia_p2p_dma_map_pages);
        nv_p2p_dma_unmap_pages = (void*)symbol_get(nvidia_p2p_dma_unmap_pages);
    } else {
        // Use kallsyms_lookup_name to find the symbols
        nv_p2p_get_pages = (void*)kallsyms_lookup_name_ptr("nvidia_p2p_get_pages");
        nv_p2p_put_pages = (void*)kallsyms_lookup_name_ptr("nvidia_p2p_put_pages");
        nv_p2p_free_page_table = (void*)kallsyms_lookup_name_ptr("nvidia_p2p_free_page_table");
        nv_p2p_dma_map_pages = (void*)kallsyms_lookup_name_ptr("nvidia_p2p_dma_map_pages");
        nv_p2p_dma_unmap_pages = (void*)kallsyms_lookup_name_ptr("nvidia_p2p_dma_unmap_pages");
    }
    
    if (!nv_p2p_get_pages || !nv_p2p_put_pages || !nv_p2p_free_page_table ||
        !nv_p2p_dma_map_pages || !nv_p2p_dma_unmap_pages) {
        printk(KERN_WARNING "Failed to resolve all NVIDIA P2P symbols\n");
        printk(KERN_WARNING "  nvidia_p2p_get_pages: %p\n", nv_p2p_get_pages);
        printk(KERN_WARNING "  nvidia_p2p_put_pages: %p\n", nv_p2p_put_pages);
        printk(KERN_WARNING "  nvidia_p2p_free_page_table: %p\n", nv_p2p_free_page_table);
        printk(KERN_WARNING "  nvidia_p2p_dma_map_pages: %p\n", nv_p2p_dma_map_pages);
        printk(KERN_WARNING "  nvidia_p2p_dma_unmap_pages: %p\n", nv_p2p_dma_unmap_pages);
        
        // Clean up any partial symbols
        if (nv_p2p_get_pages) symbol_put(nvidia_p2p_get_pages);
        if (nv_p2p_put_pages) symbol_put(nvidia_p2p_put_pages);
        if (nv_p2p_free_page_table) symbol_put(nvidia_p2p_free_page_table);
        if (nv_p2p_dma_map_pages) symbol_put(nvidia_p2p_dma_map_pages);
        if (nv_p2p_dma_unmap_pages) symbol_put(nvidia_p2p_dma_unmap_pages);
        
        nv_p2p_get_pages = NULL;
        nv_p2p_put_pages = NULL;
        nv_p2p_free_page_table = NULL;
        nv_p2p_dma_map_pages = NULL;
        nv_p2p_dma_unmap_pages = NULL;
        
        return -ENOENT;
    }
    
    printk(KERN_INFO "Successfully resolved NVIDIA P2P symbols:\n");
    printk(KERN_INFO "  nvidia_p2p_get_pages: %p\n", nv_p2p_get_pages);
    printk(KERN_INFO "  nvidia_p2p_put_pages: %p\n", nv_p2p_put_pages);
    printk(KERN_INFO "  nvidia_p2p_free_page_table: %p\n", nv_p2p_free_page_table);
    printk(KERN_INFO "  nvidia_p2p_dma_map_pages: %p\n", nv_p2p_dma_map_pages);
    printk(KERN_INFO "  nvidia_p2p_dma_unmap_pages: %p\n", nv_p2p_dma_unmap_pages);
    return 0;
}

void cleanup_nvidia_symbols(void)
{
    // Release symbol references if we got them via symbol_get
    if (nv_p2p_get_pages) {
        symbol_put(nvidia_p2p_get_pages);
        nv_p2p_get_pages = NULL;
    }
    if (nv_p2p_put_pages) {
        symbol_put(nvidia_p2p_put_pages);
        nv_p2p_put_pages = NULL;
    }
    if (nv_p2p_free_page_table) {
        symbol_put(nvidia_p2p_free_page_table);
        nv_p2p_free_page_table = NULL;
    }
    if (nv_p2p_dma_map_pages) {
        symbol_put(nvidia_p2p_dma_map_pages);
        nv_p2p_dma_map_pages = NULL;
    }
    if (nv_p2p_dma_unmap_pages) {
        symbol_put(nvidia_p2p_dma_unmap_pages);
        nv_p2p_dma_unmap_pages = NULL;
    }
}

#endif /* _CUDA */
