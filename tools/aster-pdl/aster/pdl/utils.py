from aster.pdl import ir as _ir
from aster.pdl.dialects._ods_common import (
    get_op_result_or_value,
    get_op_results_or_values,
)
import typing as _typing

################################################################################
# Type aliases
################################################################################

# Type alias for attribute constants
AttrConst: _typing.TypeAlias = _typing.Union[
    _ir.Attribute,
    _ir.Type,
    str,
    int,
    float,
    list[int],
    list[float],
    list[str],
    _typing.Mapping[str, _typing.Any],
]
# Type alias for operation or value
OpOrValue: _typing.TypeAlias = _typing.Union[_ir.OpView, _ir.Operation, _ir.Value]
# Type alias for type constants
TypeConst: _typing.TypeAlias = _typing.Union[_ir.TypeAttr, _ir.Type, str]

################################################################################
# Attr conversion utilities
################################################################################


def attr_from_str(value: str) -> _ir.Attribute:
    """Parse an MLIR Attribute from its string representation."""
    return _ir.Attribute.parse(value)


def str_attr(value: str) -> _ir.StringAttr:
    """Create an MLIR StringAttr from a string."""
    return _ir.StringAttr.get(value)


def to_attr(
    value: AttrConst,
) -> _ir.Attribute:
    """Convert a value to an MLIR Attribute."""
    if isinstance(value, _ir.Attribute):
        return value
    elif isinstance(value, _ir.Type):
        return _ir.TypeAttr.get(value)
    elif isinstance(value, int):
        return _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), value)
    elif isinstance(value, float):
        return _ir.FloatAttr.get_f32(value)
    elif isinstance(value, str):
        return attr_from_str(value)
    elif isinstance(value, _typing.Sequence):
        if all(isinstance(v, int) for v in value):
            int_type = _ir.IntegerType.get_signless(32)
            elems = [_ir.IntegerAttr.get(int_type, v) for v in value]
            return _ir.DenseI32ArrayAttr.get(elems)
        elif all(isinstance(v, float) for v in value):
            elems = [_ir.FloatAttr.get_f32(v) for v in value]
            return _ir.DenseF32ArrayAttr.get(elems)
        else:
            elems = [to_attr(v) for v in value]
            return _ir.ArrayAttr.get(elems)
    elif isinstance(value, _typing.Mapping):
        items = {k: to_attr(v) for k, v in value.items()}
        return _ir.DictAttr.get(items)
    raise ValueError("unsupported attribute type")


################################################################################
# Type conversion utilities
################################################################################


def type_from_str(value: str) -> _ir.Type:
    """Parse an MLIR Type from its string representation."""
    return _ir.Type.parse(value)


def to_type(
    value: TypeConst | str,
) -> _ir.Type:
    """Convert a value to an MLIR Type."""
    if isinstance(value, _ir.Type):
        return value
    elif isinstance(value, _ir.TypeAttr):
        return value.type
    elif isinstance(value, str):
        return type_from_str(value)
    raise ValueError("unsupported type conversion")


################################################################################
# Type and value checking utilities
################################################################################


def check_type(
    value: _typing.Optional[_typing.Any],
    types: tuple[type, ...],
):
    """Check that a value is of one of the specified types."""
    if value is None:
        return
    if not isinstance(value, types):
        type_names = ", ".join([str(t) for t in types])
        raise TypeError(f"value must be of type(s): {type_names}, got: {type(value)}")


def check_elem_types(
    value: _typing.Optional[_typing.Sequence[_typing.Any]],
    types: tuple[type, ...],
):
    """Check that all elements in a sequence are of one of the specified types."""
    if value is None:
        return
    for v in value:
        check_type(v, types)


def check_value(value: _typing.Optional[_typing.Any]):
    """Check that a value is an OpView, Operation, or Value."""
    if value is None:
        return
    check_type(value, (_ir.OpView, _ir.Operation, _ir.Value))


def isunioninstance(value: _typing.Any, types: _typing.Any) -> bool:
    """Check if a value is an instance of any of the specified types."""
    return isinstance(value, types.__args__)


################################################################################
# MLIR value helpers
################################################################################


def to_value(
    value: OpOrValue,
) -> _typing.Optional[_ir.Value]:
    """Get the MLIR Value from an OpOrValue."""
    if value is None:
        return None
    check_type(value, (_ir.OpView, _ir.Operation, _ir.Value))
    return get_op_result_or_value(value)


def value_or_passthrough(v: _typing.Optional[_typing.Any]) -> _typing.Any:
    """Get the MLIR Value from an OpOrValue or return the input as is."""
    if v is None:
        return None
    if isinstance(v, (_ir.OpView, _ir.Operation, _ir.Value)):
        return get_op_result_or_value(v)
    return v


def get_value_or_none(
    value: _typing.Optional[OpOrValue],
) -> _typing.Optional[_ir.Value]:
    """Get the MLIR Value from an OpOrValue or return None."""
    if value is None:
        return None
    return get_op_result_or_value(value)


def get_value_type_or_none(
    value: _typing.Optional[OpOrValue],
) -> _typing.Optional[_ir.Value]:
    """Get the MLIR Value from an OpOrValue or return None."""
    if value is None:
        return None
    return get_op_result_or_value(value).type
