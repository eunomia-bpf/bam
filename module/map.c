#include "map.h"
#include "list.h"
#include "ctrl.h"
#include <linux/version.h>
#include <linux/sched.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/slab.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/err.h>
#include <linux/pci.h>
#include <linux/scatterlist.h>
#include <linux/gfp.h>

#ifdef _CUDA
#include <nv-p2p.h>
#include "nvidia_wrapper.h"

struct gpu_region
{
    nvidia_p2p_page_table_t* pages;
    nvidia_p2p_dma_mapping_t** mappings;
    bool is_bypassed;  /* Flag to indicate if P2P was bypassed */
    struct sg_table* sgt;  /* Scatter-gather table for bypassed DMA mapping */
    dma_addr_t* dma_addrs;  /* DMA addresses for bypassed pages */
    struct page** bypass_pages;  /* Real pages allocated for bypassed mode */
    unsigned long n_bypass_pages;  /* Number of pages allocated for bypassed mode */
};
#endif


#define GPU_PAGE_SHIFT  16
#define GPU_PAGE_SIZE   (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK   ~(GPU_PAGE_SIZE - 1)

uint32_t max_num_ctrls = 64;


static struct map* create_descriptor(const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    unsigned long i;
    struct map* map = NULL;

    map = kvmalloc(sizeof(struct map) + n_pages * sizeof(uint64_t), GFP_KERNEL);
    if (map == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return ERR_PTR(-ENOMEM);
    }

    list_node_init(&map->list);

    map->owner = current;
    map->vaddr = vaddr;
    map->pdev = ctrl->pdev;
    map->page_size = 0;
    map->data = NULL;
    map->release = NULL;
    map->n_addrs = n_pages;


    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = 0;
    }

    return map;
}



void unmap_and_release(struct map* map)
{
    if (map == NULL)
        return;

    list_remove(&map->list);

    if (map->release != NULL && map->data != NULL)
    {
        map->release(map);
        map->release = NULL;  /* Prevent double release */
        map->data = NULL;
    }

    kvfree(map);
}



struct map* map_find(const struct list* list, u64 vaddr)
{
    const struct list_node* element;
    struct map* map = NULL;
    struct map* found = NULL;
    unsigned long flags;

    spin_lock_irqsave(&((struct list*)list)->lock, flags);

    element = list_next(&list->head);
    while (element != NULL)
    {
        map = container_of(element, struct map, list);

        if (map->owner == current)
        {
            if (vaddr >= map->vaddr && vaddr < (map->vaddr + map->n_addrs * map->page_size))
            {
                found = map;
                break;
            }
        }
        element = list_next(element);
    }

    spin_unlock_irqrestore(&list->lock, flags);

    return found;
}



struct map* find_and_remove(struct list* list, u64 vaddr)
{
    struct map* map = map_find(list, vaddr);

    if (map != NULL)
    {
        list_remove(&map->list);
        return map;
    }

    return NULL;
}



static void release_user_pages(struct map* map)
{
    struct page** pages = (struct page**) map->data;

    if (pages != NULL)
    {
        unsigned long i;
        for (i = 0; i < map->n_addrs; ++i)
        {
            if (pages[i] != NULL)
            {
                SetPageDirty(pages[i]);
                unpin_user_page(pages[i]);
            }
        }

        kfree(pages);
        map->data = NULL;

        //printk(KERN_DEBUG "Released %lu host pages\n", map->n_addrs);
    }
}



static long map_user_pages(struct map* map)
{
    long i;
    long n_pages = map->n_addrs;
    struct page** pages = NULL;

    pages = kcalloc(n_pages, sizeof(struct page*), GFP_KERNEL);
    if (pages == NULL)
    {
        printk(KERN_CRIT "Failed to allocate page array\n");
        return -ENOMEM;
    }

    map->data = pages;
    map->release = &release_user_pages;

    n_pages = pin_user_pages_fast(map->vaddr, n_pages, FOLL_WRITE | FOLL_LONGTERM, pages);
    if (n_pages <= 0)
    {
        kfree(pages);
        map->data = NULL;
        printk(KERN_ERR "get_user_pages() failed: %ld\n", n_pages);
        return n_pages;
    }
    map->n_addrs = n_pages;

    for (i = 0; i < n_pages; ++i)
    {
        dma_addr_t addr = dma_map_page(&map->pdev->dev, pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL);

        if (dma_mapping_error(&map->pdev->dev, addr))
        {
            printk(KERN_ERR "Failed to map page %ld for DMA\n", i);
            return -ENOMEM;
        }

        map->addrs[i] = addr;
    }

    return 0;
}



struct map* map_userspace(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    long err;
    struct map* md;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = PAGE_SIZE;

    err = map_user_pages(md);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu host pages starting at address %llx\n",
    //        md->n_addrs, md->vaddr);
    return md;
}



#ifdef _CUDA
static void force_release_gpu_memory(struct map* map)
{
    struct gpu_region* gd = (struct gpu_region*) map->data;
    struct list* list = map->ctrl_list;
    unsigned long flags;

    if (gd != NULL)
    {
        if (gd->is_bypassed)
        {
            /* Handle bypassed P2P cleanup */
            if (gd->sgt && list != NULL)
            {
                const struct list_node* element;
                struct ctrl* ctrl;

                spin_lock_irqsave(&list->lock, flags);
                element = list_next(&list->head);
                while (element != NULL)
                {
                    ctrl = container_of(element, struct ctrl, list);
                    if (ctrl->pdev != NULL)
                    {
                        dma_unmap_sg(&ctrl->pdev->dev, gd->sgt->sgl, gd->sgt->nents, DMA_BIDIRECTIONAL);
                        break;  /* Only need to unmap once for bypassed mode */
                    }
                    element = list_next(element);
                }
                spin_unlock_irqrestore(&list->lock, flags);

                sg_free_table(gd->sgt);
                kfree(gd->sgt);
                gd->sgt = NULL;
            }

            /* Free the real pages allocated for bypass */
            if (gd->bypass_pages)
            {
                unsigned long i;
                unsigned int order = get_order(GPU_PAGE_SIZE);
                for (i = 0; i < gd->n_bypass_pages; i++)
                {
                    if (gd->bypass_pages[i])
                        __free_pages(gd->bypass_pages[i], order);
                }
                kfree(gd->bypass_pages);
                gd->bypass_pages = NULL;
            }

            if (gd->dma_addrs)
            {
                kfree(gd->dma_addrs);
                gd->dma_addrs = NULL;
            }
        }
        else
        {
            /* Handle regular P2P cleanup */
            if (gd->mappings != NULL && list != NULL)
            {
                const struct list_node* element;
                struct ctrl* ctrl;
                uint32_t j = 0;

                spin_lock_irqsave(&list->lock, flags);
                element = list_next(&list->head);
                while (element != NULL)
                {
                    ctrl = container_of(element, struct ctrl, list);
                    if (gd->mappings[j] != NULL && ctrl->pdev != NULL)
                        nvidia_p2p_dma_unmap_pages(ctrl->pdev, gd->pages, gd->mappings[j++]);

                    element = list_next(element);
                }
                spin_unlock_irqrestore(&list->lock, flags);

                kfree(gd->mappings);
                gd->mappings = NULL;
            }

            if (gd->pages != NULL)
            {
                nvidia_p2p_free_page_table(gd->pages);
                gd->pages = NULL;
            }
        }

        kfree(gd);
        map->data = NULL;
        map->release = NULL;  /* Prevent double release */

        printk(KERN_DEBUG "Nvidia driver forcefully reclaimed %lu GPU pages\n", map->n_addrs);
    }

    /* Remove from list and free the map structure */
    list_remove(&map->list);
    kvfree(map);
}
#endif



#ifdef _CUDA
void release_gpu_memory(struct map* map)
{
    struct gpu_region* gd = (struct gpu_region*) map->data;
    struct list* list = map->ctrl_list;
    unsigned long flags;

    if (gd != NULL)
    {
        if (gd->is_bypassed)
        {
            /* Handle bypassed P2P cleanup */
            if (gd->sgt && list != NULL)
            {
                const struct list_node* element;
                struct ctrl* ctrl;

                spin_lock_irqsave(&list->lock, flags);
                element = list_next(&list->head);
                while (element != NULL)
                {
                    ctrl = container_of(element, struct ctrl, list);
                    if (ctrl->pdev != NULL)
                    {
                        dma_unmap_sg(&ctrl->pdev->dev, gd->sgt->sgl, gd->sgt->nents, DMA_BIDIRECTIONAL);
                        break;  /* Only need to unmap once for bypassed mode */
                    }
                    element = list_next(element);
                }
                spin_unlock_irqrestore(&list->lock, flags);

                sg_free_table(gd->sgt);
                kfree(gd->sgt);
                gd->sgt = NULL;
            }

            /* Free the real pages allocated for bypass */
            if (gd->bypass_pages)
            {
                unsigned long i;
                unsigned int order = get_order(GPU_PAGE_SIZE);
                for (i = 0; i < gd->n_bypass_pages; i++)
                {
                    if (gd->bypass_pages[i])
                        __free_pages(gd->bypass_pages[i], order);
                }
                kfree(gd->bypass_pages);
                gd->bypass_pages = NULL;
            }

            if (gd->dma_addrs)
            {
                kfree(gd->dma_addrs);
                gd->dma_addrs = NULL;
            }
        }
        else
        {
            /* Handle regular P2P cleanup */
            if (gd->mappings != NULL && gd->pages != NULL && list != NULL)
            {
                const struct list_node* element;
                struct ctrl* ctrl;
                uint32_t j = 0;

                spin_lock_irqsave(&list->lock, flags);
                element = list_next(&list->head);
                while (element != NULL)
                {
                    ctrl = container_of(element, struct ctrl, list);
                    if (gd->mappings[j] != NULL && ctrl->pdev != NULL)
                        nvidia_p2p_dma_unmap_pages(ctrl->pdev, gd->pages, gd->mappings[j++]);

                    element = list_next(element);
                }
                spin_unlock_irqrestore(&list->lock, flags);

                kfree(gd->mappings);
                gd->mappings = NULL;
            }

            if (gd->pages != NULL)
            {
                nvidia_p2p_put_pages(0, 0, map->vaddr, gd->pages);
                gd->pages = NULL;
            }
        }

        kfree(gd);
        map->data = NULL;

        //printk(KERN_DEBUG "Released %lu GPU pages\n", map->n_addrs);
    }
}
#endif


#ifdef _CUDA
static bool is_p2p_bypassed(struct nvidia_p2p_page_table* pages)
{
    /* Check if this is a fake page table from P2P bypass
     * The bypass module creates tables with specific patterns we can detect
     */
    if (!pages) return false;

    /* Check for bypass indicators:
     * 1. Version mismatch or unusual version number
     * 2. Page size not matching GPU page size
     * 3. Other indicators from the bypass module
     */
    if (pages->version != NVIDIA_P2P_PAGE_TABLE_VERSION)
    {
        printk(KERN_INFO "P2P BYPASS detected: unusual page table version %d\n", pages->version);
        return true;
    }

    /* Additional checks can be added here based on bypass module behavior */
    return false;
}

static int map_bypassed_gpu_memory(struct map* map, struct gpu_region* gd, struct list* list)
{
    unsigned long i;
    int err;
    struct scatterlist* sg;
    const struct list_node* element;
    struct ctrl* ctrl = NULL;
    unsigned long flags;
    struct page** pages = NULL;
    unsigned int order;
    size_t total_size;

    printk(KERN_INFO "P2P BYPASS: Allocating real pages for bypassed GPU memory\n");

    /* Calculate the order for page allocation (GPU_PAGE_SIZE = 64KB = 2^16) */
    order = get_order(GPU_PAGE_SIZE);

    /* Allocate array to store page pointers */
    pages = kcalloc(map->n_addrs, sizeof(struct page*), GFP_KERNEL);
    if (!pages)
    {
        printk(KERN_ERR "Failed to allocate page pointer array\n");
        return -ENOMEM;
    }

    /* Allocate real pages for each GPU page */
    for (i = 0; i < map->n_addrs; i++)
    {
        /* Allocate contiguous pages of GPU_PAGE_SIZE */
        pages[i] = alloc_pages(GFP_KERNEL | __GFP_ZERO, order);
        if (!pages[i])
        {
            printk(KERN_ERR "Failed to allocate page %lu (order %u)\n", i, order);
            /* Free already allocated pages */
            while (i > 0)
            {
                i--;
                __free_pages(pages[i], order);
            }
            kfree(pages);
            return -ENOMEM;
        }
        printk(KERN_DEBUG "P2P BYPASS: Allocated page %lu at %p\n", i, page_address(pages[i]));
    }

    /* Allocate scatter-gather table */
    gd->sgt = kmalloc(sizeof(struct sg_table), GFP_KERNEL);
    if (!gd->sgt)
    {
        printk(KERN_ERR "Failed to allocate sg_table\n");
        for (i = 0; i < map->n_addrs; i++)
            __free_pages(pages[i], order);
        kfree(pages);
        return -ENOMEM;
    }

    err = sg_alloc_table(gd->sgt, map->n_addrs, GFP_KERNEL);
    if (err)
    {
        printk(KERN_ERR "Failed to allocate scatter list: %d\n", err);
        kfree(gd->sgt);
        gd->sgt = NULL;
        for (i = 0; i < map->n_addrs; i++)
            __free_pages(pages[i], order);
        kfree(pages);
        return err;
    }

    /* Allocate DMA address array */
    gd->dma_addrs = kmalloc(map->n_addrs * sizeof(dma_addr_t), GFP_KERNEL);
    if (!gd->dma_addrs)
    {
        sg_free_table(gd->sgt);
        kfree(gd->sgt);
        gd->sgt = NULL;
        for (i = 0; i < map->n_addrs; i++)
            __free_pages(pages[i], order);
        kfree(pages);
        return -ENOMEM;
    }

    /* Initialize scatter list with real pages */
    sg = gd->sgt->sgl;
    for (i = 0; i < map->n_addrs; i++)
    {
        /* Set up scatter list with real allocated pages */
        sg_set_page(sg, pages[i], GPU_PAGE_SIZE, 0);
        sg = sg_next(sg);
    }

    /* Store pages array in gd for later cleanup */
    gd->bypass_pages = pages;
    gd->n_bypass_pages = map->n_addrs;

    /* Get the first controller for DMA mapping */
    spin_lock_irqsave(&list->lock, flags);
    element = list_next(&list->head);
    if (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);
    }
    spin_unlock_irqrestore(&list->lock, flags);

    if (!ctrl || !ctrl->pdev)
    {
        printk(KERN_ERR "No controller available for DMA mapping\n");
        kfree(gd->dma_addrs);
        sg_free_table(gd->sgt);
        kfree(gd->sgt);
        gd->sgt = NULL;
        gd->dma_addrs = NULL;
        for (i = 0; i < map->n_addrs; i++)
            __free_pages(pages[i], order);
        kfree(pages);
        return -ENODEV;
    }

    /* Map the scatter-gather list for DMA */
    err = dma_map_sg(&ctrl->pdev->dev, gd->sgt->sgl, gd->sgt->nents, DMA_BIDIRECTIONAL);
    if (err <= 0)
    {
        printk(KERN_ERR "Failed to map scatter-gather list for DMA: %d\n", err);
        kfree(gd->dma_addrs);
        sg_free_table(gd->sgt);
        kfree(gd->sgt);
        gd->sgt = NULL;
        gd->dma_addrs = NULL;
        for (i = 0; i < map->n_addrs; i++)
            __free_pages(pages[i], order);
        kfree(pages);
        return -ENOMEM;
    }

    /* Fill in the DMA addresses */
    sg = gd->sgt->sgl;
    for (i = 0; i < map->n_addrs; i++)
    {
        gd->dma_addrs[i] = sg_dma_address(sg);
        map->addrs[i] = gd->dma_addrs[i];
        printk(KERN_DEBUG "P2P BYPASS: Page %lu: DMA addr = 0x%llx\n", i, gd->dma_addrs[i]);
        sg = sg_next(sg);
    }

    printk(KERN_INFO "P2P BYPASS: Successfully mapped %lu GPU pages with real pages for DMA\n", map->n_addrs);
    return 0;
}
#endif


#ifdef _CUDA
int map_gpu_memory(struct map* map, struct list* list)
{
    unsigned long i;
    uint32_t j;
    int err;
    struct gpu_region* gd;
    const struct list_node* element;
    struct ctrl* ctrl;

    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);
    if (gd == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return -ENOMEM;
    }

    /* Initialize gpu_region structure */
    gd->pages = NULL;
    gd->mappings = NULL;
    gd->is_bypassed = false;
    gd->sgt = NULL;
    gd->dma_addrs = NULL;
    gd->bypass_pages = NULL;
    gd->n_bypass_pages = 0;

    gd->mappings = (nvidia_p2p_dma_mapping_t**) kmalloc(sizeof(nvidia_p2p_dma_mapping_t*) * max_num_ctrls, GFP_KERNEL);
    if (gd->mappings == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        kfree(gd);
        return -ENOMEM;
    }
    for (j = 0; j < max_num_ctrls; j++)
        gd->mappings[j] = NULL;

    map->page_size = GPU_PAGE_SIZE;
    map->data = gd;
    map->release = release_gpu_memory;

    printk(KERN_INFO "Attempting nvidia_p2p_get_pages...\n");

    /* For NVIDIA P2P: p2p_token is typically the GPU address, va_space is the PID */
    err = nvidia_p2p_get_pages(map->vaddr, current->tgid, map->vaddr, GPU_PAGE_SIZE * map->n_addrs, &gd->pages,
            (void (*)(void*)) force_release_gpu_memory, map);

    if (err != 0)
    {
        printk(KERN_ERR "nvidia_p2p_get_pages() failed: %d\n", err);
        kfree(gd->mappings);
        kfree(gd);
        return err;
    }

    /* Check if P2P was bypassed */
    if (is_p2p_bypassed(gd->pages))
    {
        printk(KERN_WARNING "P2P BYPASS detected: Using alternative DMA mapping\n");
        gd->is_bypassed = true;

        /* Use alternative DMA mapping for bypassed P2P */
        err = map_bypassed_gpu_memory(map, gd, list);
        if (err != 0)
        {
            printk(KERN_ERR "Failed to map bypassed GPU memory: %d\n", err);
            if (gd->pages)
                nvidia_p2p_free_page_table(gd->pages);
            kfree(gd->mappings);
            kfree(gd);
            /* Prevent double-free by clearing pointers */
            map->data = NULL;
            map->release = NULL;
            return err;
        }
    }
    else
    {
        /* Normal P2P path */
        printk(KERN_INFO "P2P: Using standard NVIDIA P2P DMA mapping\n");

        /* Map pages for each controller with proper locking */
        j = 0;
        {
            unsigned long flags;

            spin_lock_irqsave(&list->lock, flags);
            element = list_next(&list->head);

            while (element != NULL)
            {
                ctrl = container_of(element, struct ctrl, list);

                /* We need to release the lock before calling nvidia functions */
                spin_unlock_irqrestore(&list->lock, flags);

                err = nvidia_p2p_dma_map_pages(ctrl->pdev, gd->pages, gd->mappings + (j++));
                if (err != 0)
                {
                    printk(KERN_ERR "nvidia_p2p_dma_map_pages() failed: %d\n", err);
                    /* Clean up already mapped pages */
                    while (j > 0)
                    {
                        j--;
                        if (gd->mappings[j])
                            nvidia_p2p_dma_unmap_pages(ctrl->pdev, gd->pages, gd->mappings[j]);
                    }
                    nvidia_p2p_put_pages(0, 0, map->vaddr, gd->pages);
                    kfree(gd->mappings);
                    kfree(gd);
                    /* Prevent double-free by clearing pointers */
                    map->data = NULL;
                    map->release = NULL;
                    return err;
                }

                if (j == 1) {
                    for (i = 0; i < map->n_addrs; ++i)
                    {
                        map->addrs[i] = gd->mappings[0]->dma_addresses[i];
                    }
                }

                /* Re-acquire lock and get next element */
                spin_lock_irqsave(&list->lock, flags);
                element = list_next(element);
            }

            spin_unlock_irqrestore(&list->lock, flags);
        }

        /* Verify page count */
        if (map->n_addrs != gd->pages->entries)
        {
            printk(KERN_WARNING "Requested %lu GPU pages, but only got %u\n", map->n_addrs, gd->pages->entries);
            map->n_addrs = gd->pages->entries;
        }
    }

    printk(KERN_INFO "Successfully mapped %lu GPU pages\n", map->n_addrs);
    return 0;
}
#endif



#ifdef _CUDA
struct map* map_device_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages, struct list* ctrl_list)
{
    int err;
    struct map* md = NULL;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & GPU_PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = GPU_PAGE_SIZE;
    md->ctrl_list = ctrl_list;
    err = map_gpu_memory(md, ctrl_list);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu GPU pages starting at address %llx\n",
    //        md->n_addrs, md->vaddr);
    return md;
}
#else



struct map* map_device_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages, struct list* ctrl_list)
{
    return ERR_PTR(-ENOTSUPP);
}
#endif