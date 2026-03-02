// server.js - OpenAI to NVIDIA NIM API Proxy (thinking mode fixed)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware - increase body size for large chat history
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ limit: '10mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Show reasoning in output
const SHOW_REASONING = false;

// Thinking mode enabled
const ENABLE_THINKING_MODE = true;

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking' // thinking mode enabled
};

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Determine NIM model
    let nimModel = MODEL_MAPPING[model] || model;

    // Build NIM request with thinking mode properly applied
    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
      stream: stream ?? false,
      ...(ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : {})
    };

    // Send request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      // Streaming mode
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n'); // correct split
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) {
            res.write(line + '\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));

            if (data.choices?.[0]?.delta) {
              const delta = data.choices[0].delta;
              let content = delta.content ?? '';
              const reasoning = delta.reasoning_content;

              if (SHOW_REASONING && reasoning) {
                content = (!reasoningStarted ? '<think>\n' + reasoning : reasoning) + (content ? '\n</think>\n\n' + content : '');
                reasoningStarted = !!content.includes('<think>');
              }

              delta.content = content;
              delete delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            res.write(line + '\n');
          }
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => { console.error('Stream error:', err); res.end(); });

    } else {
      // Non-streaming
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map((choice) => {
          let content = choice.message?.content ?? '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            content = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + content;
          }
          return { index: choice.index, message: { role: choice.message.role, content }, finish_reason: choice.finish_reason };
        }),
        usage: response.data.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: { message: error.message || 'Internal server error', type: 'invalid_request_error', code: error.response?.status || 500 }
    });
  }
});

// Homepage route
app.get('/', (req, res) => res.send('NVIDIA NIM Proxy is running.'));

// Catch-all 404
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

// Start server
app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
