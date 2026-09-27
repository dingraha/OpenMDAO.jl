using SysInfo: SysInfo
using ReTestItems
using OpenMDAOCore: OpenMDAOCore

runtests(
    OpenMDAOCore;
    # nworkers = max(1, SysInfo.ncores() - 1),
    nworkers = SysInfo.ncores(),
    nworker_threads = 1,
    testitem_timeout = 600,
)
