# llm_provider.py

import os
import requests
import openai
import importlib
import rospy

class BaseLLMProvider:
    def chat(self, system_message: str, user_prompt: str) -> str:
        raise NotImplementedError
# -------------------------
# Provider Implementations
# -------------------------
class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name, max_tokens, temperature):
        openai.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system_message, user_prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            rospy.logerr(f"OpenAI API error: {e}")
            return "I'm sorry, I couldn't process that request."

class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name, max_tokens, temperature):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system_message, user_prompt):
        try:
            endpoint = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            rospy.logerr(f"DeepSeek API error: {e}")
            return "I'm sorry, I couldn't process that request."

class LlamaCppProvider(BaseLLMProvider):
    def __init__(self, api_key_unused, model_name_unused, max_tokens, temperature):
        self.endpoint = "http://localhost:8000/completion"  # Adjust to the llama.cpp server URL
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system_message, user_prompt):
        try:
            payload = {
                "prompt": f"{system_message}\n{user_prompt}",
                "n_predict": self.max_tokens,
                "temperature": self.temperature,
            }
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("content", "").strip()
        except Exception as e:
            rospy.logerr(f"Llama.cpp API error: {e}")
            return "I'm sorry, I couldn't process that request."

class ClaudeProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name, max_tokens, temperature):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system_message, user_prompt):
        try:
            endpoint = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "user", "content": f"{system_message}\n{user_prompt}"}
                ]
            }
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"].strip()
        except Exception as e:
            rospy.logerr(f"Claude API error: {e}")
            return "I'm sorry, I couldn't process that request."

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name_unused, max_tokens_unused, temperature_unused):
        self.api_key = api_key

    def chat(self, system_message, user_prompt):
        try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": system_message},
                        {"text": user_prompt}
                    ]
                }]
            }
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            rospy.logerr(f"Gemini API error: {e}")
            return "I'm sorry, I couldn't process that request."

# -------------------------
# Factory Loader
# -------------------------

class LLMProviderFactory:
    provider_classes = {
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider,
        "llama.cpp": LlamaCppProvider,
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
    }

    @classmethod
    def create_provider(cls, provider_name, api_key, model_name, max_tokens, temperature):
        if provider_name not in cls.provider_classes:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
        provider_class = cls.provider_classes[provider_name]
        return provider_class(api_key, model_name, max_tokens, temperature)
