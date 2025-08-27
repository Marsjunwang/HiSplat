"""
导入工具模块 - 提供智能的导入机制，兼容不同运行环境
"""

import sys
import os
from pathlib import Path

def get_xfeat_model():
    """获取XFeatModel类"""
    try:
        # 首先尝试相对导入（包内运行时）
        from .model import XFeatModel
        return XFeatModel
    except (ImportError, ValueError):
        # 相对导入失败，尝试绝对导入（独立运行时）
        current_dir = Path(__file__).parent.absolute()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        try:
            from model import XFeatModel
            return XFeatModel
        except ImportError as e:
            raise ImportError(f"无法导入XFeatModel: {e}")

def get_interpolate_sparse2d():
    """获取InterpolateSparse2d类"""
    try:
        # 首先尝试相对导入（包内运行时）
        from .interpolator import InterpolateSparse2d
        return InterpolateSparse2d
    except (ImportError, ValueError):
        # 相对导入失败，尝试绝对导入（独立运行时）
        current_dir = Path(__file__).parent.absolute()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        try:
            from interpolator import InterpolateSparse2d
            return InterpolateSparse2d
        except ImportError as e:
            raise ImportError(f"无法导入InterpolateSparse2d: {e}")

def get_all_from_model():
    """获取model模块的所有内容"""
    try:
        # 首先尝试相对导入（包内运行时）
        import importlib
        from . import model
        return model
    except (ImportError, ValueError):
        # 相对导入失败，尝试绝对导入（独立运行时）
        current_dir = Path(__file__).parent.absolute()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        try:
            import model
            return model
        except ImportError as e:
            raise ImportError(f"无法导入model模块: {e}")