"""
配置文件管理模块
获取config.json中的配置，提供配置访问接口
"""

import json
import os


config_base={
    "coze_api_base": "https://api.coze.cn",
    "coze_token": "",
    "coze_bot_id": "",
    "bot_type": "coze",
    "businessHours": {
        "start": "08:00",
        "end": "23:00"
    },
    "openai_api_base": "https://api.openai.com/v1",
    "openai_api_key": "",
    "openai_model": "gpt-3.5-turbo"
}

class Config:
    def __init__(self, config_path='config.json'):
        """初始化配置类"""
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {self.config_path} 不存在，正在创建默认配置文件")
            # 使用config_base创建配置文件
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_base, f, ensure_ascii=False, indent=4)
                print(f"已创建默认配置文件：{self.config_path}")
                return config_base.copy()
            except Exception as e:
                print(f"创建配置文件失败: {e}")
                return config_base.copy()
        except json.JSONDecodeError:
            print(f"错误: 配置文件 {self.config_path} 格式不正确")
            return config_base.copy()
    
    def get(self, key, default=None):
        """获取配置项"""
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        """支持使用字典方式访问配置"""
        return self.config[key]
    
    def __contains__(self, key):
        """支持使用 in 操作符检查配置项"""
        return key in self.config
    
    def reload(self):
        """重新加载配置文件"""
        self.config = self._load_config()
        return self.config
    
    def set(self, key, value, save=False):
        """
        设置配置项
        
        Args:
            key: 配置项键名
            value: 配置项值
            save: 是否立即保存到文件，默认为False
        """
        self.config[key] = value
        if save:
            self.save()
        return value
    
    def save(self):
        """将当前配置保存到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False
    
    def update(self, config_dict, save=False):
        """
        批量更新配置
        
        Args:
            config_dict: 包含多个配置项的字典
            save: 是否立即保存到文件，默认为False
        """
        self.config.update(config_dict)
        if save:
            self.save()
        return self.config

# 创建全局配置实例
config = Config()
