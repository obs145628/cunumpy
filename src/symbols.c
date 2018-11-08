#define _GNU_SOURCE

#include "symbols.h"
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <string.h>

static const char* lib_name = "pyhack.cpython-36m-x86_64-linux-gnu.so";
static const char* g_sym_name;
static void* g_sym;

static int substr_at_end(const char* s, const char* sub)
{
  size_t len_s = strlen(s);
  size_t len_sub = strlen(sub);
  
  if (len_sub > len_s)
    return 0;


  const char* s_sub = s + len_s - len_sub;
  
  return strcmp(sub, s_sub) == 0;
  
}


static int callback_find_sym(struct dl_phdr_info* info, size_t size, void* data)
{
  (void) size;
  (void) data;
  
  if (*(info->dlpi_name) == 0)
    return 0;

  if (substr_at_end(info->dlpi_name, lib_name))
    return 0;
  
  void* dl_lib = dlopen(info->dlpi_name, RTLD_LAZY);
  void* sym = dlsym(dl_lib, g_sym_name);

  if (sym)
  {
    printf("Symbol %s found in %s\n", g_sym_name, info->dlpi_name);
    g_sym = sym;
    return 1;
  }
  
  return 0;
}

void* find_symbol(const char* name)
{
  g_sym_name = name;
  g_sym = NULL;
  dl_iterate_phdr(callback_find_sym, NULL);
  return g_sym;
}
