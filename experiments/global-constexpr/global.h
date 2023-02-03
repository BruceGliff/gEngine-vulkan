
#include <array>
#include <typeindex>
#include <vector>

struct TypeDesc {

  std::type_index Ty;
  void *Owner{nullptr};

  TypeDesc(std::type_index TyIn) : Ty{TyIn} {}

  template <typename T> TypeDesc static create() { return TypeDesc(typeid(T)); }
};

class global {
public:
  std::vector<TypeDesc> Map;

  constexpr global(std::initializer_list<TypeDesc> list) : Map(list) {}
};

// Interface:
// At some module:
//  creates array of necessary types.
//    G.requires<Image, int, std::vector>();
//
// G = global::getInstance();
// G.register<Image>( pointer );
//
//  somewhere: G.get<Image>();
