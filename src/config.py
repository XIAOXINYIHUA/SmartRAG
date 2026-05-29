"""统一配置中心 - 多平台 Provider 预设"""

import os

PROVIDERS: dict[str, dict] = {
    "dashscope": {
        "name": "阿里云 DashScope（通义千问）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "llm_models": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-plus-latest"],
        "embedding_models": ["text-embedding-v3", "text-embedding-v2", "text-embedding-v1"],
        "default_llm": "qwen-plus",
        "default_embedding": "text-embedding-v3",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "llm_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "embedding_models": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        "default_llm": "gpt-4o-mini",
        "default_embedding": "text-embedding-3-small",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "llm_models": ["deepseek-chat", "deepseek-reasoner"],
        "embedding_models": [],  # DeepSeek 没有 embedding 服务
        "default_llm": "deepseek-chat",
        "default_embedding": None,
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "siliconflow": {
        "name": "硅基流动 (SiliconFlow)",
        "base_url": "https://api.siliconflow.cn/v1",
        "llm_models": [
            "Qwen/Qwen3-235B-A22B",
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen2.5-72B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Pro/ZhipuAI/GLM-4-Plus",
        ],
        "embedding_models": [
            "BAAI/bge-large-zh-v1.5",
            "BAAI/bge-m3",
            "netease-youdao/bce-embedding-base_v1",
        ],
        "default_llm": "Qwen/Qwen3-235B-A22B",
        "default_embedding": "BAAI/bge-large-zh-v1.5",
        "api_key_env": "SILICONFLOW_API_KEY",
    },
    "zhipu": {
        "name": "智谱 AI (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "llm_models": ["glm-4-flash", "glm-4-plus", "glm-4-long"],
        "embedding_models": ["embedding-2"],
        "default_llm": "glm-4-flash",
        "default_embedding": "embedding-2",
        "api_key_env": "ZHIPU_API_KEY",
    },
    "ollama": {
        "name": "Ollama（本地模型）",
        "base_url": "http://localhost:11434/v1",
        "llm_models": ["qwen3", "llama3", "deepseek-r1", "mistral"],
        "embedding_models": ["nomic-embed-text", "bge-m3"],
        "default_llm": "qwen3",
        "default_embedding": "nomic-embed-text",
        "api_key_env": None,  # Ollama 无需 API Key
    },
}

DEFAULT_PROVIDER = "dashscope"

# 没有 embedding 能力的平台，回退到此平台做 embedding
EMBEDDING_FALLBACK_PROVIDER = "dashscope"


def get_provider_config(provider_id: str) -> dict:
    """获取平台完整配置"""
    if provider_id not in PROVIDERS:
        raise ValueError(f"不支持的平台: {provider_id}，可用: {list(PROVIDERS)}")
    return PROVIDERS[provider_id]


def get_api_key(provider_id: str) -> str:
    """获取平台的 API Key，未配置则抛出明确错误"""
    cfg = get_provider_config(provider_id)
    env_var = cfg.get("api_key_env")
    if env_var is None:
        return "ollama"
    key = os.getenv(env_var, "")
    if not key:
        raise ValueError(
            f"未设置 {env_var}，请在 .env 文件中配置或在侧边栏输入该平台的 API Key"
        )
    return key


def get_embedding_provider(provider_id: str) -> str:
    """返回实际用于 embedding 的平台 ID"""
    cfg = get_provider_config(provider_id)
    if cfg["embedding_models"]:
        return provider_id
    return EMBEDDING_FALLBACK_PROVIDER


def list_providers() -> list[tuple[str, str]]:
    """返回所有平台 (id, name) 列表"""
    return [(k, v["name"]) for k, v in PROVIDERS.items()]
