"""Small Windows-only helper for pinning the hand overlay to every virtual desktop.

Windows does not expose a supported Win32 API for pinning a window to all virtual
 desktops. The shell's IVirtualDesktopPinnedApps COM service provides the same
operation used by Explorer's "Show this window on all desktops" command. Calls are
best-effort and deliberately fail closed on unsupported Windows versions.
"""
from __future__ import annotations

import ctypes
import sys
import uuid
from ctypes import wintypes


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def parse(cls, value: str) -> "_GUID":
        parsed = uuid.UUID(value)
        return cls(
            parsed.time_low,
            parsed.time_mid,
            parsed.time_hi_version,
            (ctypes.c_ubyte * 8).from_buffer_copy(parsed.bytes[8:]),
        )


def _com_call(pointer, slot, argtypes, *args):
    """Call an HRESULT-returning COM vtable method using its zero-based slot."""
    factory = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    method = factory(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])
    return method(pointer, *args)


def _release(pointer) -> None:
    if pointer:
        try:
            _com_call(pointer, 2, [])  # IUnknown::Release
        except Exception:
            pass


def pin_window_on_all_virtual_desktops(hwnd: int) -> bool:
    """Pin a top-level HWND to all Windows virtual desktops, returning success.

    This is a Windows-only, best-effort operation. It does not activate the window
    or change its z-order. If the shell's private service is not available, callers
    can continue with the normal overlay behaviour.
    """
    if not sys.platform.startswith("win") or not hwnd:
        return False

    # Shell service and interface IDs used by the Windows virtual desktop manager.
    clsid_immersive_shell = _GUID.parse("C2F03A33-21F5-47FA-B4BB-156362A2F239")
    iid_service_provider = _GUID.parse("6D5140C1-7436-11CE-8034-00AA006009FA")
    iid_application_view_collection = _GUID.parse("1841C6D7-4F9D-42C0-AF41-8747538F10E5")
    clsid_pinned_apps = _GUID.parse("B5A399E7-1C87-46B8-88E9-FC5747B171BD")
    iid_pinned_apps = _GUID.parse("4CE81583-1E4C-4632-A621-07A53543148F")

    ole32 = ctypes.WinDLL("ole32")
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoUninitialize.argtypes = []
    ole32.CoUninitialize.restype = None
    ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(_GUID), ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p),
    ]
    ole32.CoCreateInstance.restype = ctypes.c_long

    # COINIT_APARTMENTTHREADED; tolerate a thread already initialized by Qt/another
    # library in a different apartment, but only balance successful initialization.
    init_hr = int(ole32.CoInitializeEx(None, 0x2))
    changed_mode = init_hr == -2147417850  # RPC_E_CHANGED_MODE
    if init_hr < 0 and not changed_mode:
        return False
    should_uninitialize = init_hr >= 0

    provider = ctypes.c_void_p()
    views = ctypes.c_void_p()
    pinned_apps = ctypes.c_void_p()
    view = ctypes.c_void_p()
    success = False
    try:
        # CLSCTX_ALL = in-proc server, handler, local server, and remote server.
        hr = int(ole32.CoCreateInstance(
            ctypes.byref(clsid_immersive_shell), None, 0x17,
            ctypes.byref(iid_service_provider), ctypes.byref(provider),
        ))
        if hr < 0 or not provider.value:
            return False

        query_service_args = [ctypes.POINTER(_GUID), ctypes.POINTER(_GUID),
                              ctypes.POINTER(ctypes.c_void_p)]
        hr = int(_com_call(
            provider, 3, query_service_args,
            ctypes.byref(iid_application_view_collection),
            ctypes.byref(iid_application_view_collection), ctypes.byref(views),
        ))  # IServiceProvider::QueryService
        if hr < 0 or not views.value:
            return False

        hr = int(_com_call(
            provider, 3, query_service_args,
            ctypes.byref(clsid_pinned_apps), ctypes.byref(iid_pinned_apps),
            ctypes.byref(pinned_apps),
        ))
        if hr < 0 or not pinned_apps.value:
            return False

        hwnd_type = wintypes.HWND
        hr = int(_com_call(
            views, 6, [hwnd_type, ctypes.POINTER(ctypes.c_void_p)],
            hwnd_type(int(hwnd)), ctypes.byref(view),
        ))  # IApplicationViewCollection::GetViewForHwnd
        if hr < 0 or not view.value:
            return False

        is_pinned = ctypes.c_int32(0)
        hr = int(_com_call(
            pinned_apps, 6, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)],
            view, ctypes.byref(is_pinned),
        ))  # IVirtualDesktopPinnedApps::IsViewPinned
        if hr < 0:
            return False
        if is_pinned.value:
            return True

        hr = int(_com_call(
            pinned_apps, 7, [ctypes.c_void_p], view,
        ))  # IVirtualDesktopPinnedApps::PinView
        if hr < 0:
            return False
        # Do not report success just because the private shell call returned a
        # non-error HRESULT: verify Explorer now considers this exact view pinned.
        confirmed = ctypes.c_int32(0)
        verify_hr = int(_com_call(
            pinned_apps, 6, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)],
            view, ctypes.byref(confirmed),
        ))  # IVirtualDesktopPinnedApps::IsViewPinned
        success = verify_hr >= 0 and bool(confirmed.value)
        return success
    except Exception:
        return False
    finally:
        _release(view)
        _release(pinned_apps)
        _release(views)
        _release(provider)
        if should_uninitialize:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass
