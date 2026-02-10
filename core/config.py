"""配置管理模块 - 处理缓存目录等持久化配置"""

import os
import sys
import json
from pathlib import Path
from typing import Optional


def get_executable_dir() -> str:
    """获取可执行文件所在目录"""
    if getattr(sys, 'frozen', False):
        # 打包后运行：返回exe所在目录
        return os.path.dirname(sys.executable)
    else:
        # 源代码运行：返回main.py所在目录
        import __main__
        if hasattr(__main__, '__file__'):
            return os.path.dirname(os.path.abspath(__main__.__file__))
        return os.getcwd()


class Config:
    """配置管理器 - 管理缓存目录、数据库路径等配置"""
    
    def __init__(self):
        self._exe_dir = get_executable_dir()
        self._config_file = os.path.join(self._exe_dir, "faceseeker_config.json")
        self._cache_dir: Optional[str] = None
        self._load_config()
    
    def _load_config(self):
        """从配置文件加载配置"""
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._cache_dir = data.get('cache_dir')
            except Exception as e:
                print(f"警告：加载配置文件失败: {e}")
                self._cache_dir = None
        
        # 如果没有配置缓存目录，使用默认值（可执行文件目录）
        if not self._cache_dir:
            self._cache_dir = self._exe_dir
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            data = {
                'cache_dir': self._cache_dir
            }
            with open(self._config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告：保存配置文件失败: {e}")
    
    @property
    def cache_dir(self) -> str:
        """缓存目录路径"""
        return self._cache_dir
    
    @cache_dir.setter
    def cache_dir(self, path: str):
        """设置缓存目录"""
        # 确保路径存在
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self._cache_dir = os.path.abspath(path)
        self._save_config()
    
    @property
    def database_path(self) -> str:
        """数据库文件完整路径"""
        return os.path.join(self._cache_dir, "faceseeker.db")
    
    @property
    def log_path(self) -> str:
        """日志文件完整路径"""
        return os.path.join(self._cache_dir, "faceseeker.log")
    
    @property
    def executable_dir(self) -> str:
        """可执行文件所在目录（用于读取模型等资源）"""
        return self._exe_dir
    
    def get_resource_path(self, relative_path: str) -> str:
        """获取资源文件路径（打包后从临时目录读取，源代码运行从项目目录读取）"""
        if getattr(sys, 'frozen', False):
            # 打包后：资源文件在Nuitka/PyInstaller的临时目录
            base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else self._exe_dir
        else:
            # 源代码运行：资源文件在项目目录
            import __main__
            if hasattr(__main__, '__file__'):
                base_path = os.path.dirname(os.path.abspath(__main__.__file__))
            else:
                base_path = os.getcwd()
        return os.path.join(base_path, relative_path)


# 全局配置实例
_global_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config
