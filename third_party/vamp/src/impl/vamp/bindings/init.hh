#include <nanobind/nanobind.h>

namespace vamp::binding
{
    void init_environment(nanobind::module_ &pymodule);
    void init_settings(nanobind::module_ &pymodule);
    void init_fetch(nanobind::module_ &pymodule);
}  // namespace vamp::binding
