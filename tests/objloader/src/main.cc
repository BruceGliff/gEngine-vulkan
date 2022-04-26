/*

Test program launches with output: Tinybj test.

Dependencies: tinyobjloader

*/
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>

int main() {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                        "NonExistingPath"))
    std::cout << "TinyObj test.\n";

  return 0;
}