// server.js - OpenAI to NVIDIA NIM API Proxy (safe, working version)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Increase max payload size
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(cors());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 Toggles
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = true; // thinking mode enabled safely
const MAX_REASONING_TOKENS = 356;

// Model mapping
const MODEL_MAPPING = {
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking', // auto map Gemini-Pro
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct'
};

// Max tokens for context safety
const MAX_CONTEXT_TOKENS = 32000;

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NVIDIA NIM Proxy' });
});

// List models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(m => ({
    id: m, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Utility: rough token count
function countTokens(messages) {
  return messages.reduce((sum, msg) => sum + Math.ceil(msg.content.length / 4), 0);
}

// Trim messages to fit context
function trimMessages(messages, maxTokens) {
  let tokenCount = 0;
  const trimmed = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const msgTokens = Math.ceil(msg.content.length / 4);
    if (tokenCount + msgTokens > maxTokens) break;
    tokenCount += msgTokens;
    trimmed.unshift(msg);
  }
  return trimmed;
}

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    let nimModel = MODEL_MAPPING[model] || model;

    // Trim messages for safe context
    const safeMessages = trimMessages(messages, MAX_CONTEXT_TOKENS);
    const safeMaxTokens = Math.min(max_tokens || 2000, MAX_CONTEXT_TOKENS - countTokens(safeMessages));

    // Build request
    const nimRequest = {
      model: nimModel,
      messages: safeMessages,
      temperature: temperature || 0.6,
      max_tokens: safeMaxTokens,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      // Streaming
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      let buffer = '';
      response.data.on('data', chunk => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        lines.forEach(line => { res.write(line + '\n'); });
      });
      response.data.on('end', () => res.end());
      response.data.on('error', err => { console.error(err); res.end(); });
    } else {
      // Non-streaming
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(choice => ({
          index: choice.index,
          message: { role: choice.message.role, content: choice.message.content },
          finish_reason: choice.finish_reason
        })),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }

  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(err.response?.status || 500).json({
      error: { message: err.message || 'Internal server error', type: 'invalid_request_error', code: err.response?.status || 500 }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
});
