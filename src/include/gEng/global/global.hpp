// File contains some template function description of class GlbManager.

namespace gEng {

template <typename T, typename... Args>
T &GlbManager::registerEntity(Args &&...args) {
  std::type_index const TyIdx{typeid(T)};

  auto FindIt = Regs.find(TyIdx);
  if (FindIt == Regs.end()) {
    bool IsInserted = true;
    std::tie(FindIt, IsInserted) = Regs.emplace(
        TyIdx, static_cast<void *>(new T{std::forward<Args>(args)...}));
    assert(IsInserted && "Inserting already existing entity.");
  } else
    // TODO logs this.
    std::cerr << "Entity already exists.\n";

  return *castIt<T>(FindIt);
}

template <typename T> T *GlbManager::getEntityIfPossible() const {
  auto FindIt = Regs.find(std::type_index{typeid(T)});
  if (FindIt == Regs.end())
    return nullptr;
  return castIt<T>(FindIt);
}

template <typename T> T &GlbManager::getEntity() const {
  T *Ent = getEntityIfPossible<T>();
  if (!Ent)
    throw std::runtime_error{"Getting unexisting entity."};
  return *Ent;
}

template <typename T> T *GlbManager::castIt(auto const &It) const {
  return static_cast<T *>(It->second);
}

} // namespace gEng
