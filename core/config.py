"""配置管理模块 - 资源文件与数据文件分离管理

资源文件（模型等）：自动释放到打包临时目录或项目目录，不可配置
数据文件（数据库、日志）：可由用户选择存放在 AppData 默认目录或自定义目录
配置文件本身始终存在 AppData 目录下
"""

import os
import sys
import json
from typing import Optional


# ---- 常量 ----

APP_NAME = "FaceAtlas"


# ---- 辅助函数 ----

def _get_appdata_base() -> str:
    """获取 AppData 下的应用目录（Windows: %APPDATA%/FaceAtlas）"""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.path.join(os.path.expanduser("~"), ".config")
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _get_resource_base() -> str:
    """获取资源文件基目录（模型等只读资源）

    打包后：PyInstaller 的 _MEIPASS 临时目录
    源代码运行：项目目录
    """
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        import __main__
        if hasattr(__main__, "__file__"):
            return os.path.dirname(os.path.abspath(__main__.__file__))
        return os.getcwd()


class Config:
    """配置管理器

    - 资源文件（模型）：通过 get_resource_path() 访问，路径固定不可配置
    - 数据文件（db / log）：通过 data_dir 属性访问，用户可选择目录
    - 配置文件本身存储在 AppData 目录下，始终固定
    """

    # 数据文件名
    DB_FILENAME = "FaceAtlas.db"
    LOG_FILENAME = "FaceAtlas.log"

    def __init__(self):
        # AppData 目录（固定）
        self._appdata_dir = _get_appdata_base()

        # 配置文件放在 AppData 下
        self._config_file = os.path.join(self._appdata_dir, "config.json")

        # 资源文件基目录（固定）
        self._resource_base = _get_resource_base()

        # 数据目录（用户可配置）
        self._data_dir: Optional[str] = None
        self._data_dir_configured: bool = False

        self._load_config()

    # ================================================
    # 配置文件读写
    # ================================================

    def _load_config(self):
        """从 AppData/config.json 加载配置"""
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._data_dir = data.get("data_dir") or None
                    self._data_dir_configured = data.get("data_dir_configured", False)
            except Exception as e:
                print(f"警告：加载配置文件失败: {e}")

    def _save_config(self):
        """保存配置到 AppData/config.json"""
        try:
            data = {
                "data_dir": self._data_dir,
                "data_dir_configured": self._data_dir_configured,
            }
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告：保存配置文件失败: {e}")

    # ================================================
    # 数据目录
    # ================================================

    @property
    def default_data_dir(self) -> str:
        """默认数据目录（AppData 下）"""
        return self._appdata_dir

    @property
    def data_dir(self) -> str:
        """当前生效的数据目录"""
        d = self._data_dir if self._data_dir else self._appdata_dir
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def is_data_dir_configured(self) -> bool:
        """用户是否已经配置过数据目录"""
        return self._data_dir_configured

    def set_data_dir(self, path: Optional[str]):
        """设置数据目录

        Args:
            path: 自定义路径。传 None 或空字符串表示使用默认 AppData 目录。
        """
        if path:
            path = os.path.abspath(path)
            os.makedirs(path, exist_ok=True)
            self._data_dir = path
        else:
            self._data_dir = None
        self._data_dir_configured = True
        self._save_config()

    # ---- 兼容旧属性名 ----

    @property
    def cache_dir(self) -> str:
        """兼容旧代码：等同于 data_dir"""
        return self.data_dir

    @cache_dir.setter
    def cache_dir(self, path: str):
        self.set_data_dir(path)

    # ================================================
    # 数据文件路径
    # ================================================

    @property
    def database_path(self) -> str:
        """数据库文件完整路径"""
        return os.path.join(self.data_dir, self.DB_FILENAME)

    @property
    def log_path(self) -> str:
        """日志文件完整路径"""
        return os.path.join(self.data_dir, self.LOG_FILENAME)

    # ================================================
    # 资源文件路径
    # ================================================

    def get_resource_path(self, relative_path: str) -> str:
        """获取只读资源文件路径（模型等）

        打包后从 _MEIPASS 读取，源代码运行从项目目录读取。
        """
        return os.path.join(self._resource_base, relative_path)

    @property
    def executable_dir(self) -> str:
        """可执行文件所在目录"""
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return self._resource_base

    # ================================================
    # 数据文件检查
    # ================================================

    @staticmethod
    def check_existing_data(directory: str) -> dict:
        """检查指定目录下是否存在已有数据文件

        Returns:
            dict: {"db": bool, "log": bool, "db_size": int, "log_size": int}
        """
        result = {"db": False, "log": False, "db_size": 0, "log_size": 0}
        db_file = os.path.join(directory, Config.DB_FILENAME)
        log_file = os.path.join(directory, Config.LOG_FILENAME)
        if os.path.isfile(db_file):
            result["db"] = True
            result["db_size"] = os.path.getsize(db_file)
        if os.path.isfile(log_file):
            result["log"] = True
            result["log_size"] = os.path.getsize(log_file)
        return result


# ---- 全局单例 ----

_global_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config
