client<llm> LlamaVisionGroqClient11b {
  provider openai-generic
  options {
    base_url "https://api.groq.com/openai/v1"
    api_key env.GROQ_API_KEY
    model "llama-3.2-11b-vision-preview"
  }
}

client<llm> LlamaVisionGroqClient {
  provider openai-generic
  options {
    base_url "https://api.groq.com/openai/v1"
    api_key env.GROQ_API_KEY
    model "llama-3.2-90b-vision-preview"
  }
}

client<llm> OpenAIClient {
    provider openai
    options {
        model gpt-4o
        api_key env.OPENAI_API_KEY
        supports_streaming false 
    }
}

client<llm> AnthropicClient {
  provider anthropic
  options {
    model claude-3-7-sonnet-20250219
    api_key env.ANTHROPIC_API_KEY
  }
}

client<llm> GeminiFlashClient {
  provider google-ai
  options {
    model "gemini-2.0-flash"
    api_key env.GEMINI_API_KEY
  }
}