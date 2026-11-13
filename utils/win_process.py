"""Windows 进程工具：枚举 / 读命令行 / 强杀进程树，全部走 Win32 API。

为什么不用 ``taskkill``、``powershell``、``netstat`` 这些系统程序：

1. 本程序是 GUI 程序（打包时 console=False），没有控制台。每次 spawn 一个
   控制台子程序，Windows 都会给它新建一个黑窗——真机表现为「一打开就闪
   一下命令行窗口」。
2. 打包后的程序把 ``_internal`` 目录塞进了 PATH，系统自带程序启动时会加载
   到冲突 DLL，直接报 0xc0000142「应用程序无法正常启动」，清理随之失效。

改用 kernel32 / ntdll 的接口后既不弹窗，也不受 PATH 影响。
"""
import ctypes
import sys
from ctypes import wintypes

IS_WINDOWS = sys.platform == "win32"

_IS_64 = sys.maxsize > 2 ** 32

PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010
TH32CS_SNAPPROCESS = 0x00000002


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


if IS_WINDOWS:
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    # 64 位下句柄必须声明成 HANDLE，否则默认 c_int 会截断
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.TerminateProcess.restype = wintypes.BOOL
    _kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _kernel32.ReadProcessMemory.restype = wintypes.BOOL
    _kernel32.ReadProcessMemory.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
    ]

    _ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    _ntdll.NtQueryInformationProcess.argtypes = [
        wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p,
        wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
    ]


def iter_processes():
    """枚举进程，yield (pid, 父pid, exe 名)。"""
    if not IS_WINDOWS:
        return
    snapshot = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snapshot:
        return
    invalid = ctypes.c_void_p(-1).value
    if snapshot == invalid:
        return
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = _kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            yield (
                int(entry.th32ProcessID),
                int(entry.th32ParentProcessID),
                str(entry.szExeFile),
            )
            ok = _kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    except Exception:
        return
    finally:
        _kernel32.CloseHandle(snapshot)


def _read_memory(handle, address, size):
    if not address or not size:
        return None
    buf = ctypes.create_string_buffer(size)
    done = ctypes.c_size_t(0)
    ok = _kernel32.ReadProcessMemory(
        handle, ctypes.c_void_p(address), buf, size, ctypes.byref(done)
    )
    if not ok:
        return None
    return buf.raw


def _read_pointer(handle, address):
    size = 8 if _IS_64 else 4
    raw = _read_memory(handle, address, size)
    if not raw:
        return None
    value = int.from_bytes(raw, "little")
    if not value:
        return None
    return value


def process_commandline(pid):
    """读进程完整命令行。读不到（权限/进程已退）返回 None。

    路径：OpenProcess → NtQueryInformationProcess 拿 PEB →
    PEB->ProcessParameters（64 位 0x20 / 32 位 0x10）→
    RTL_USER_PROCESS_PARAMETERS->CommandLine（64 位 0x70 / 32 位 0x40）。
    """
    if not IS_WINDOWS:
        return None
    handle = _kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, int(pid)
    )
    if not handle:
        return None
    try:
        info = ctypes.create_string_buffer(48)
        returned = wintypes.ULONG(0)
        status = _ntdll.NtQueryInformationProcess(
            handle, 0, info, 48, ctypes.byref(returned)
        )
        if status != 0:
            return None
        peb = int.from_bytes(info.raw[8:16], "little")
        if not peb:
            return None
        params_addr = peb + (0x20 if _IS_64 else 0x10)
        params = _read_pointer(handle, params_addr)
        if not params:
            return None
        cl_offset = 0x70 if _IS_64 else 0x40
        raw = _read_memory(handle, params + cl_offset, 16 if _IS_64 else 8)
        if not raw:
            return None
        length = int.from_bytes(raw[0:2], "little")
        if _IS_64:
            buffer_ptr = int.from_bytes(raw[8:16], "little")
        else:
            buffer_ptr = int.from_bytes(raw[4:8], "little")
        if not length or not buffer_ptr:
            return None
        data = _read_memory(handle, buffer_ptr, length)
        if not data:
            return None
        return data.decode("utf-16-le", errors="replace")
    except Exception:
        return None
    finally:
        _kernel32.CloseHandle(handle)


def kill_pid(pid):
    """强杀单个进程。已退出返回 False，不抛异常。"""
    if not IS_WINDOWS:
        return False
    handle = _kernel32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
    if not handle:
        return False
    try:
        ok = _kernel32.TerminateProcess(handle, 1)
        return bool(ok)
    except Exception:
        return False
    finally:
        _kernel32.CloseHandle(handle)


def child_pids(parent_pid, pairs=None):
    """递归收集所有子孙进程 pid。pairs 可传入已取好的 (pid, ppid) 快照。"""
    if pairs is None:
        pairs = []
        for pid, ppid, _name in iter_processes():
            pairs.append((pid, ppid))
    children = []
    frontier = [int(parent_pid)]
    while frontier:
        current = frontier.pop()
        for pid, ppid in pairs:
            if ppid == current and pid not in children:
                children.append(pid)
                frontier.append(pid)
    return children


def kill_process_tree(pid):
    """强杀进程树（含子孙），返回被杀的进程数。

    Chromium 的渲染/GPU 子进程未必带 ``--user-data-dir``，按命令行匹配会漏，
    所以已经有 pid 时优先杀整棵树。
    """
    if not IS_WINDOWS or not pid:
        return 0
    killed = 0
    try:
        targets = child_pids(pid)
    except Exception:
        targets = []
    targets.append(int(pid))
    for target in targets:
        if kill_pid(target):
            killed += 1
    return killed


def kill_by_commandline(marker, exe_names):
    """强杀命令行包含 ``marker`` 的指定进程（大小写不敏感），返回杀掉的数量。

    用途：本程序启动的浏览器都带 ``--user-data-dir=<自家数据目录>``，
    用户自己开的浏览器用的是默认目录，用这个标记就能精确区分——绝不按
    进程名全量杀，避免误杀用户正在用的浏览器。
    """
    if not IS_WINDOWS or not marker:
        return 0
    wanted = set()
    for name in exe_names:
        wanted.add(name.lower())

    needle = marker.lower().replace("/", "\\")
    matched = []
    tree_seeds = []
    pairs = []
    for pid, ppid, name in iter_processes():
        pairs.append((pid, ppid))
        if name.lower() not in wanted:
            continue
        cmd = process_commandline(pid)
        if not cmd:
            continue
        if needle not in cmd.lower().replace("/", "\\"):
            continue
        matched.append(pid)
        tree_seeds.append(pid)

    # 先把所有目标和它们的子孙一次算全，再统一杀。
    # 顺序很重要：先杀父再去找子，父子关系就不在快照里了。
    targets_to_kill = []
    for pid in matched:
        if pid not in targets_to_kill:
            targets_to_kill.append(pid)
    for pid in tree_seeds:
        try:
            children = child_pids(pid, pairs)
        except Exception:
            children = []
        for child in children:
            if child not in targets_to_kill:
                targets_to_kill.append(child)

    killed = 0
    for pid in targets_to_kill:
        if kill_pid(pid):
            killed += 1
    return killed
