import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    """保存配置到YAML文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def print_config(config: Dict[str, Any]):
    """打印配置内容"""
    print("=" * 50)
    print("训练配置:")
    print("=" * 50)
    print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    print("=" * 50)