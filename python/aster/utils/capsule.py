"""PyCapsule pointer wrapping/unwrapping helpers."""


def wrap_pointer_in_capsule(ptr):
    """Wrap a pointer in a PyCapsule.

    Args:
        ptr: ctypes pointer value (c_void_p or ctypes.addressof result)

    Returns:
        PyCapsule containing the pointer.
    """
    import ctypes
    from ctypes import pythonapi, py_object, c_void_p, c_char_p

    PyCapsule_New = pythonapi.PyCapsule_New
    PyCapsule_New.restype = py_object
    PyCapsule_New.argtypes = [c_void_p, c_char_p, c_void_p]
    return PyCapsule_New(ptr, b"nb_handle", None)


def unwrap_pointer_from_capsule(capsule):
    """Extract a pointer value from a PyCapsule.

    Args:
        capsule: PyCapsule containing a pointer

    Returns:
        Raw pointer value (c_void_p).
    """
    import ctypes

    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.restype = ctypes.c_void_p
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return PyCapsule_GetPointer(ctypes.py_object(capsule), b"nb_handle")
