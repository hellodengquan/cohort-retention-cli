import re
import ast
import operator
from typing import Any, Dict, Optional, Callable, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


class DSLError(Exception):
    pass


class DSLErrorRecovery:
    def __init__(
        self,
        enabled: bool = True,
        coalesce_on_error: bool = True,
        default_numeric: float = 0.0,
        default_string: str = "",
        default_datetime: Optional[str] = None,
    ):
        self.enabled = enabled
        self.coalesce_on_error = coalesce_on_error
        self.default_numeric = default_numeric
        self.default_string = default_string
        self.default_datetime = default_datetime
        self.warnings: List[str] = []

    def recover(self, error: Exception, dtype_hint: Optional[str] = None) -> Any:
        self.warnings.append(str(error))
        if not self.enabled:
            raise
        if self.coalesce_on_error:
            if dtype_hint == "numeric":
                return self.default_numeric
            elif dtype_hint == "string":
                return self.default_string
            elif dtype_hint == "datetime":
                return pd.NaT if self.default_datetime is None else pd.to_datetime(self.default_datetime)
            else:
                return self._infer_default(error)
        raise

    def _infer_default(self, error: Exception) -> Any:
        error_str = str(error).lower()
        if any(k in error_str for k in ["numeric", "number", "int", "float", "division"]):
            return self.default_numeric
        elif any(k in error_str for k in ["string", "str", "text"]):
            return self.default_string
        elif any(k in error_str for k in ["datetime", "date", "time", "timestamp"]):
            return pd.NaT
        return None


class DSLParser:
    _BINARY_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.And: lambda a, b: a & b if isinstance(a, (pd.Series, np.ndarray)) else a and b,
        ast.Or: lambda a, b: a | b if isinstance(a, (pd.Series, np.ndarray)) else a or b,
    }

    _UNARY_OPS = {
        ast.Not: operator.not_,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @staticmethod
    def _infer_dtype(values: List[Any]) -> str:
        has_numeric = False
        has_string = False
        has_datetime = False
        has_bool = False

        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, (pd.Series, np.ndarray)):
                dtype = v.dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    has_numeric = True
                elif pd.api.types.is_string_dtype(dtype):
                    has_string = True
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    has_datetime = True
                elif pd.api.types.is_bool_dtype(dtype):
                    has_bool = True
            elif isinstance(v, bool):
                has_bool = True
            elif isinstance(v, (int, float, np.number)):
                has_numeric = True
            elif isinstance(v, str):
                has_string = True
            elif isinstance(v, (datetime, pd.Timestamp)):
                has_datetime = True

        if has_numeric and not has_string and not has_datetime:
            return "numeric"
        if has_string and not has_datetime:
            return "string"
        if has_datetime:
            return "datetime"
        if has_bool:
            return "bool"
        return "unknown"

    @classmethod
    def _coalesce(cls, *args, error_recovery: Optional[DSLErrorRecovery] = None):
        try:
            if any(isinstance(a, (pd.Series, np.ndarray)) for a in args):
                result = None
                for a in args:
                    try:
                        if result is None:
                            result = pd.Series(a) if not isinstance(a, (pd.Series, np.ndarray)) else a
                        else:
                            fill_val = a
                            result = result.fillna(fill_val)
                    except Exception as e:
                        if error_recovery and error_recovery.enabled:
                            continue
                        raise
                if result is None:
                    dtype = cls._infer_dtype(list(args))
                    if error_recovery:
                        return error_recovery.recover(ValueError("所有参数均无效"), dtype)
                    return pd.Series([np.nan] * max(len(a) for a in args if isinstance(a, (pd.Series, np.ndarray))))
                return result
            else:
                for a in args:
                    try:
                        if pd.notna(a):
                            return a
                    except Exception:
                        if error_recovery and error_recovery.enabled:
                            continue
                        raise
                dtype = cls._infer_dtype(list(args))
                if error_recovery:
                    return error_recovery.recover(ValueError("所有参数均无效"), dtype)
                return None
        except Exception as e:
            if error_recovery and error_recovery.enabled:
                dtype = cls._infer_dtype(list(args))
                return error_recovery.recover(e, dtype)
            raise

    @staticmethod
    def _if_else(cond, true_val, false_val):
        if isinstance(cond, (pd.Series, np.ndarray)):
            return pd.Series(np.where(cond, true_val, false_val))
        else:
            return true_val if cond else false_val

    @staticmethod
    def _date_diff_days(d1, d2):
        d1_pd = pd.to_datetime(d1)
        d2_pd = pd.to_datetime(d2)
        if isinstance(d1, (pd.Series, np.ndarray)) or isinstance(d2, (pd.Series, np.ndarray)):
            return (d2_pd - d1_pd).dt.days
        else:
            return (d2_pd - d1_pd).days

    @staticmethod
    def _date_add_days(d, n):
        d_pd = pd.to_datetime(d)
        days = int(n)
        if isinstance(d, (pd.Series, np.ndarray, pd.DatetimeIndex)):
            return d_pd + pd.Timedelta(days=days)
        else:
            return d_pd + timedelta(days=days)

    @classmethod
    def _min(cls, *args, error_recovery: Optional[DSLErrorRecovery] = None):
        try:
            if len(args) == 0:
                raise ValueError("min 需要至少一个参数")
            if len(args) == 1:
                arg = args[0]
                if isinstance(arg, (pd.Series, np.ndarray, list)):
                    if len(arg) == 0:
                        if error_recovery and error_recovery.enabled:
                            return error_recovery.recover(ValueError("空序列"), "numeric")
                        return np.nan
                    return np.nanmin(arg) if isinstance(arg, (pd.Series, np.ndarray)) else min(arg)
                elif hasattr(arg, 'min'):
                    return arg.min()
                elif isinstance(arg, (int, float, np.number)):
                    return arg
                else:
                    return min(arg)
            else:
                if all(isinstance(a, (pd.Series, np.ndarray)) for a in args):
                    return np.minimum.reduce(args)
                elif any(isinstance(a, (pd.Series, np.ndarray)) for a in args):
                    arrays = [pd.Series(a) if not isinstance(a, (pd.Series, np.ndarray)) else a for a in args]
                    return np.minimum.reduce(arrays)
                else:
                    return min(args)
        except Exception as e:
            if error_recovery and error_recovery.enabled:
                return error_recovery.recover(e, "numeric")
            raise

    @classmethod
    def _max(cls, *args, error_recovery: Optional[DSLErrorRecovery] = None):
        try:
            if len(args) == 0:
                raise ValueError("max 需要至少一个参数")
            if len(args) == 1:
                arg = args[0]
                if isinstance(arg, (pd.Series, np.ndarray, list)):
                    if len(arg) == 0:
                        if error_recovery and error_recovery.enabled:
                            return error_recovery.recover(ValueError("空序列"), "numeric")
                        return np.nan
                    return np.nanmax(arg) if isinstance(arg, (pd.Series, np.ndarray)) else max(arg)
                elif hasattr(arg, 'max'):
                    return arg.max()
                elif isinstance(arg, (int, float, np.number)):
                    return arg
                else:
                    return max(arg)
            else:
                if all(isinstance(a, (pd.Series, np.ndarray)) for a in args):
                    return np.maximum.reduce(args)
                elif any(isinstance(a, (pd.Series, np.ndarray)) for a in args):
                    arrays = [pd.Series(a) if not isinstance(a, (pd.Series, np.ndarray)) else a for a in args]
                    return np.maximum.reduce(arrays)
                else:
                    return max(args)
        except Exception as e:
            if error_recovery and error_recovery.enabled:
                return error_recovery.recover(e, "numeric")
            raise

    @staticmethod
    def _sum(*args):
        if len(args) == 1:
            if isinstance(args[0], (pd.Series, np.ndarray, list)):
                return np.sum(args[0])
            elif hasattr(args[0], 'sum'):
                return args[0].sum()
        return np.sum(args)

    @staticmethod
    def _count(*args):
        if len(args) == 1 and isinstance(args[0], (pd.Series, np.ndarray, list)):
            return len(args[0])
        elif len(args) == 1 and hasattr(args[0], '__len__'):
            return len(args[0])
        return len(args)

    _FUNCTIONS = {
        "min": _min.__func__,
        "max": _max.__func__,
        "abs": np.abs,
        "sum": _sum.__func__,
        "mean": np.mean,
        "median": np.median,
        "count": _count.__func__,
        "coalesce": _coalesce.__func__,
        "if_else": _if_else.__func__,
        "date_diff_days": _date_diff_days.__func__,
        "date_add_days": _date_add_days.__func__,
        "contains": lambda s, pat: s.str.contains(str(pat)) if isinstance(s, pd.Series) else str(pat) in str(s),
        "startswith": lambda s, pat: s.str.startswith(str(pat)) if isinstance(s, pd.Series) else str(s).startswith(str(pat)),
        "endswith": lambda s, pat: s.str.endswith(str(pat)) if isinstance(s, pd.Series) else str(s).endswith(str(pat)),
        "lower": lambda s: s.str.lower() if isinstance(s, pd.Series) else str(s).lower(),
        "upper": lambda s: s.str.upper() if isinstance(s, pd.Series) else str(s).upper(),
        "year": lambda d: pd.to_datetime(d).dt.year if isinstance(d, (pd.Series, pd.DatetimeIndex)) else pd.to_datetime(d).year,
        "month": lambda d: pd.to_datetime(d).dt.month if isinstance(d, (pd.Series, pd.DatetimeIndex)) else pd.to_datetime(d).month,
        "day": lambda d: pd.to_datetime(d).dt.day if isinstance(d, (pd.Series, pd.DatetimeIndex)) else pd.to_datetime(d).day,
        "quarter": lambda d: pd.to_datetime(d).dt.quarter if isinstance(d, (pd.Series, pd.DatetimeIndex)) else pd.to_datetime(d).quarter,
        "isnull": pd.isna,
        "notnull": pd.notna,
        "to_date": pd.to_datetime,
        "floor_day": lambda d: pd.to_datetime(d).dt.floor("D"),
        "floor_week": lambda d: pd.to_datetime(d).dt.to_period("W").dt.to_timestamp(),
        "floor_month": lambda d: pd.to_datetime(d).dt.to_period("M").dt.to_timestamp(),
        "floor_quarter": lambda d: pd.to_datetime(d).dt.to_period("Q").dt.to_timestamp(),
    }

    @classmethod
    def _parse_node(
        cls,
        node: ast.AST,
        context: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        error_recovery: Optional[DSLErrorRecovery] = None,
    ) -> Any:
        try:
            if isinstance(node, ast.Expression):
                return cls._parse_node(node.body, context, df, error_recovery)

            elif isinstance(node, ast.Constant):
                return node.value

            elif isinstance(node, ast.Name):
                if node.id in context:
                    return context[node.id]
                if df is not None and node.id in df.columns:
                    return df[node.id]
                raise DSLError(f"未找到变量或列: {node.id}")

            elif isinstance(node, ast.Attribute):
                base = cls._parse_node(node.value, context, df, error_recovery)
                attr = node.attr
                if isinstance(base, pd.Series):
                    if pd.api.types.is_string_dtype(base):
                        try:
                            if hasattr(base.str, attr):
                                return getattr(base.str, attr)
                        except (AttributeError, ValueError):
                            pass
                    if pd.api.types.is_datetime64_any_dtype(base):
                        try:
                            if hasattr(base.dt, attr):
                                return getattr(base.dt, attr)
                        except (AttributeError, ValueError):
                            pass
                if hasattr(base, attr):
                    return getattr(base, attr)
                raise DSLError(f"对象没有属性: {attr}")

            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type in cls._BINARY_OPS:
                    left = cls._parse_node(node.left, context, df, error_recovery)
                    right = cls._parse_node(node.right, context, df, error_recovery)
                    try:
                        return cls._BINARY_OPS[op_type](left, right)
                    except Exception as e:
                        if error_recovery and error_recovery.enabled:
                            dtype = cls._infer_dtype([left, right])
                            return error_recovery.recover(e, dtype)
                        raise
                raise DSLError(f"不支持的二元运算符: {op_type.__name__}")

            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type in cls._UNARY_OPS:
                    operand = cls._parse_node(node.operand, context, df, error_recovery)
                    return cls._UNARY_OPS[op_type](operand)
                raise DSLError(f"不支持的一元运算符: {op_type.__name__}")

            elif isinstance(node, ast.BoolOp):
                op_type = type(node.op)
                if op_type == ast.And:
                    result = None
                    for value in node.values:
                        val = cls._parse_node(value, context, df, error_recovery)
                        result = val if result is None else result & val if isinstance(result, (pd.Series, np.ndarray)) else result and val
                    return result
                elif op_type == ast.Or:
                    result = None
                    for value in node.values:
                        val = cls._parse_node(value, context, df, error_recovery)
                        result = val if result is None else result | val if isinstance(result, (pd.Series, np.ndarray)) else result or val
                    return result
                raise DSLError(f"不支持的布尔运算符: {op_type.__name__}")

            elif isinstance(node, ast.Compare):
                left = cls._parse_node(node.left, context, df, error_recovery)
                for op, comp in zip(node.ops, node.comparators):
                    right = cls._parse_node(comp, context, df, error_recovery)
                    op_type = type(op)
                    if op_type in cls._BINARY_OPS:
                        result = cls._BINARY_OPS[op_type](left, right)
                        left = result
                    else:
                        raise DSLError(f"不支持的比较运算符: {op_type.__name__}")
                return left

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in cls._FUNCTIONS:
                        args = [cls._parse_node(arg, context, df, error_recovery) for arg in node.args]
                        kwargs = {kw.arg: cls._parse_node(kw.value, context, df, error_recovery) for kw in node.keywords}
                        func = cls._FUNCTIONS[func_name]
                        if func_name in ("min", "max", "coalesce"):
                            if "error_recovery" not in kwargs:
                                kwargs["error_recovery"] = error_recovery
                        return func(*args, **kwargs)
                    raise DSLError(f"不支持的函数: {func_name}")
                else:
                    func_obj = cls._parse_node(node.func, context, df, error_recovery)
                    if callable(func_obj):
                        args = [cls._parse_node(arg, context, df, error_recovery) for arg in node.args]
                        kwargs = {kw.arg: cls._parse_node(kw.value, context, df, error_recovery) for kw in node.keywords}
                        return func_obj(*args, **kwargs)
                    raise DSLError(f"不支持的函数调用")

            elif isinstance(node, ast.Subscript):
                base = cls._parse_node(node.value, context, df, error_recovery)

                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    slice_ = node.slice.value
                else:
                    slice_ = cls._parse_node(node.slice, context, df, error_recovery)

                if isinstance(base, pd.DataFrame):
                    if isinstance(slice_, str):
                        return base[slice_]
                    elif isinstance(slice_, (pd.Series, np.ndarray)):
                        if slice_.dtype == bool:
                            return base[slice_]
                        else:
                            return base.iloc[slice_]
                    elif isinstance(slice_, (int, slice)):
                        return base.iloc[slice_]
                    else:
                        return base[slice_]
                elif isinstance(base, pd.Series):
                    if isinstance(slice_, (int, slice)):
                        return base.iloc[slice_]
                    else:
                        return base[slice_]
                else:
                    return base[slice_]

            elif isinstance(node, ast.Slice):
                lower = cls._parse_node(node.lower, context, df, error_recovery) if node.lower else None
                upper = cls._parse_node(node.upper, context, df, error_recovery) if node.upper else None
                step = cls._parse_node(node.step, context, df, error_recovery) if node.step else None
                return slice(lower, upper, step)

            elif isinstance(node, ast.List):
                return [cls._parse_node(el, context, df, error_recovery) for el in node.elts]

            elif isinstance(node, ast.Tuple):
                return tuple(cls._parse_node(el, context, df, error_recovery) for el in node.elts)

            elif isinstance(node, ast.Dict):
                return {
                    cls._parse_node(key, context, df, error_recovery): cls._parse_node(value, context, df, error_recovery)
                    for key, value in zip(node.keys, node.values)
                }

            raise DSLError(f"不支持的语法节点: {type(node).__name__}")
        except Exception as e:
            if error_recovery and error_recovery.enabled:
                return error_recovery.recover(e)
            raise

    @classmethod
    def evaluate(
        cls,
        expression: str,
        context: Optional[Dict[str, Any]] = None,
        df: Optional[pd.DataFrame] = None,
        error_recovery: Optional[DSLErrorRecovery] = None,
    ) -> Any:
        if context is None:
            context = {}

        try:
            tree = ast.parse(expression, mode="eval")
            return cls._parse_node(tree, context, df, error_recovery)
        except SyntaxError as e:
            raise DSLError(f"表达式语法错误: {e}")
        except DSLError:
            raise
        except Exception as e:
            if error_recovery and error_recovery.enabled:
                return error_recovery.recover(e)
            raise DSLError(f"表达式计算失败: {e}")


class CohortDSLEngine:
    def __init__(self, dsl_config: "CohortDSLConfig"):
        self.config = dsl_config
        self.error_recovery = DSLErrorRecovery(
            enabled=dsl_config.error_recovery,
            coalesce_on_error=dsl_config.coalesce_on_error,
            default_numeric=dsl_config.default_numeric,
            default_string=dsl_config.default_string,
            default_datetime=dsl_config.default_datetime,
        )

    def apply_filter(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if not self.config.filter_expression:
            return df
        mask = DSLParser.evaluate(
            self.config.filter_expression,
            context,
            df,
            error_recovery=self.error_recovery,
        )
        if isinstance(mask, (pd.Series, np.ndarray)):
            return df[mask.fillna(False) if isinstance(mask, pd.Series) else mask]
        elif mask:
            return df
        else:
            return df.iloc[0:0]

    def compute_cohort_time(
        self,
        events_df: pd.DataFrame,
        user_col: str,
        time_col: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        user_groups = events_df.groupby(user_col)

        if self.config.cohort_expression:
            def _compute_cohort(group):
                ctx = {"group": group, "user_id": group[user_col].iloc[0]}
                if context:
                    ctx.update(context)
                return DSLParser.evaluate(
                    self.config.cohort_expression,
                    ctx,
                    group,
                    error_recovery=self.error_recovery,
                )

            try:
                cohort_times = user_groups.apply(_compute_cohort, include_groups=False).reset_index()
            except TypeError:
                cohort_times = user_groups.apply(_compute_cohort).reset_index()
            cohort_times.columns = [user_col, "cohort_time"]
        else:
            cohort_times = events_df.sort_values(time_col).groupby(user_col).first().reset_index()
            cohort_times = cohort_times[[user_col, time_col]].rename(columns={time_col: "cohort_time"})

        cohort_times["cohort_time"] = pd.to_datetime(cohort_times["cohort_time"])
        return cohort_times

    def compute_segment(
        self,
        df: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.Series:
        if not self.config.segment_expression:
            return pd.Series(["default"] * len(df), index=df.index, name="segment")

        result = DSLParser.evaluate(
            self.config.segment_expression,
            context,
            df,
            error_recovery=self.error_recovery,
        )
        if isinstance(result, pd.Series):
            return result.rename("segment")
        else:
            return pd.Series([result] * len(df), index=df.index, name="segment")

    def validate_expressions(self) -> Tuple[bool, List[str]]:
        errors = []

        test_context = {
            "user_id": "test_user",
            "event_type": "order",
            "amount": 100.0,
        }

        for expr_name, expr in [
            ("cohort_expression", self.config.cohort_expression),
            ("filter_expression", self.config.filter_expression),
            ("segment_expression", self.config.segment_expression),
        ]:
            if expr:
                try:
                    ast.parse(expr, mode="eval")
                except SyntaxError as e:
                    errors.append(f"{expr_name} 语法错误: {e}")

        return len(errors) == 0, errors


__all__ = ["DSLError", "DSLParser", "CohortDSLEngine"]
