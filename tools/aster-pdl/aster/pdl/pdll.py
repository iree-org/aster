from aster.pdl import ir as _ir
from aster.pdl.utils import *
from aster.pdl.dialects.builtin import ModuleOp as _ModuleOp
from dataclasses import dataclass as _dataclass

import aster.pdl.dialects.pdl as _pdl
import typing as _typing
from typing import Optional as _Opt, Sequence as _Seq

# Type aliases for better readability
PDLAttrArg: _typing.TypeAlias = _typing.Union[AttrConst, OpOrValue]
PDLTypeArg: _typing.TypeAlias = _typing.Union[TypeConst, OpOrValue]


def _to_attr(value: _Opt[AttrConst]) -> _Opt[_ir.Attribute]:
    """Convert a value to an MLIR Attribute or None."""
    return None if value is None else to_attr(value)


def _to_type(value: _Opt[TypeConst]) -> _Opt[_ir.Type]:
    """Convert a value to an MLIR Type or None."""
    return None if value is None else to_type(value)


def _get_pdl_attr(
    type: _Opt[PDLTypeArg],
) -> _Opt[_ir.Value]:
    """Get a pdl type or None."""
    if type is None:
        return None
    type = value_or_passthrough(type)
    if isinstance(type, _ir.Value):
        return type
    return pdl_attr(type)


def _get_pdl_type(
    type: _Opt[PDLTypeArg],
) -> _Opt[_ir.Value]:
    """Get a pdl type or None."""
    if type is None:
        return None
    type = value_or_passthrough(type)
    if isinstance(type, _ir.Value):
        return type
    return pdl_type(type)


def pdl_apply_native_constraint(
    results: _Seq[TypeConst],
    name: str,
    arguments: _Seq[OpOrValue],
    is_negated: bool = False,
) -> _Seq[_ir.Value]:
    """Apply a native constraint in PDL."""
    return _pdl.ApplyNativeConstraintOp(
        [to_type(r) for r in results], name, arguments, is_negated
    ).results


def pdl_apply_native_rewrite(
    results: _Seq[TypeConst],
    name: str,
    arguments: _Seq[OpOrValue],
) -> _Seq[_ir.Value]:
    """Apply a native rewrite in PDL."""
    return _pdl.ApplyNativeRewriteOp(
        [to_type(r) for r in results], name, arguments
    ).results


def pdl_attr(
    value: _Opt[AttrConst] = None,
    type: _Opt[PDLTypeArg] = None,
) -> _ir.Value:
    """Create a pdl.attr operation to define an attribute handle."""
    return _pdl.AttributeOp(valueType=_get_pdl_type(type), value=_to_attr(value)).result


def pdl_erase(op: OpOrValue) -> _ir.Value:
    """Create a pdl.erase operation to mark an operation for erasure."""
    return _pdl.EraseOp(value_or_passthrough(op)).result


def pdl_operand(
    type: _Opt[PDLTypeArg] = None,
) -> _ir.Value:
    """Create a pdl.operand operation to capture a single operand."""
    return _pdl.OperandOp(type=_get_pdl_type(type)).result


def pdl_operands(
    types: _Opt[_Seq[PDLTypeArg]] = None,
) -> _ir.Value:
    """Create a pdl.operands operation to capture a range of operands."""
    if types is not None:
        types = pdl_types(types)
    return _pdl.OperandsOp(types=types).result


def pdl_operation(
    name: _Opt[str] = None,
    args: _Opt[_Seq[OpOrValue]] = None,
    attributes: _Opt[dict[str, PDLAttrArg]] = None,
    types: _Opt[_Seq[PDLTypeArg]] = None,
) -> _ir.Value:
    """Create a pdl.operation to define or match an operation."""
    if attributes:
        attributes = {k: _get_pdl_attr(v) for k, v in attributes.items()}
    if types:
        types = [_get_pdl_type(type) for type in types]
    return _pdl.OperationOp(
        name=name,
        args=args,
        attributes=attributes,
        types=types,
    ).result


def pdl_range(
    arguments: _Seq[OpOrValue],
    is_type_range: bool = False,
) -> _ir.Value:
    """Create a pdl.range operation to construct a range of PDL entities."""
    result_type = (
        _pdl.RangeType.get(_pdl.TypeType.get())
        if is_type_range
        else _pdl.RangeType.get(_pdl.ValueType.get())
    )
    return _pdl.RangeOp(result_type, arguments=arguments).result


def pdl_replace(
    op: OpOrValue,
    with_op: _Opt[OpOrValue] = None,
    with_values: _Opt[_Seq[OpOrValue]] = None,
) -> _pdl.ReplaceOp:
    """Create a pdl.replace operation to mark an operation as replaced."""
    return _pdl.ReplaceOp(
        op=op,
        with_op=with_op,
        with_values=with_values,
    )


def pdl_result(
    parent: OpOrValue,
    index: int,
) -> _ir.Value:
    """Create a pdl.result operation to extract a single result from an operation."""
    return _pdl.ResultOp(parent=parent, index=index).result


def pdl_results(
    parent: OpOrValue,
    index: _Opt[int] = None,
) -> _ir.Value:
    """Create a pdl.results operation to extract a result group from an operation."""
    return _pdl.ResultsOp(parent=parent, index=index).result


def pdl_rewrite(
    name: str,
    args: _Seq[OpOrValue],
    root: _Opt[OpOrValue] = None,
):
    _pdl.RewriteOp(root, name, args)


def pdl_type(type: _Opt[TypeConst] = None) -> _ir.Value:
    return _pdl.TypeOp(type=_to_type(type)).result


def pdl_types(types: _Opt[_Seq[TypeConst]] = None) -> _ir.Value:
    """Create a pdl.types operation to define a range of type handles."""
    return _pdl.TypesOp(
        constantTypes=[to_type(t) for t in types] if types is not None else None
    ).result


def pattern(benefit: int = 0):
    """Decorator to define a PDL pattern."""

    def _decorator(func: _typing.Callable[[], None]):
        def wrapper():
            assert callable(func), "pattern decorator can only be applied to functions"
            assert func.__name__, "pattern function must have a name"
            pattern = _pdl.PatternOp(name=func.__name__, benefit=benefit)
            pattern.nonmaterializable = True
            with _ir.InsertionPoint(pattern.body):
                func()
            return pattern

        if _ir.InsertionPoint.current is not None:
            wrapper()
        return wrapper

    return _decorator


def pattern_rewrite(
    root: _Opt[OpOrValue] = None,
):
    """Decorator to define a PDL rewrite."""

    def _decorator(func: _typing.Callable[[], None]):
        def wrapper():
            assert callable(func), "rewrite decorator can only be applied to functions"
            rewrite = _pdl.RewriteOp(root)
            with _ir.InsertionPoint(rewrite.body):
                func()
            return rewrite

        wrapper()
        return wrapper

    return _decorator
