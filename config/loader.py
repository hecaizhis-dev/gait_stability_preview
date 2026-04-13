"""配置加载模块，从YAML读取参数，支持项目根目录解析"""
import os
import yaml
from pathlib import Path

def _get_project_root():
    """从当前文件往上找到项目根目录（含config的目录）"""
    current=Path(__file__).resolve()
    return current.parent.parent

def load_config(config_name='default.yaml'):
    """加载配置文件，返回合并后的配置字典"""
    root=_get_project_root()
    config_path=root/'config'/config_name
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path,encoding='utf-8') as f:
        cfg=yaml.safe_load(f)
    cfg['_root']=str(root)
    return cfg
