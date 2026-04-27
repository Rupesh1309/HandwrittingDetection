require('dotenv').config();
const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── API Key Pools ─────────────────────────────────────────────────────────────
// Supports GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3 (primary)
// Also supports GROK_API_KEY as fallback (if credits are added later)
function buildPool(prefix) {
  const keys = [];
  const k1 = process.env[prefix];
  if (k1 && !k1.includes('your_')) keys.push(k1);
  let i = 2;
  while (true) {
    const k = process.env[`${prefix}_${i}`];
    if (!k) break;
    if (!k.includes('your_')) keys.push(k);
    i++;
  }
  return keys;
}

let geminiKeys = buildPool('GEMINI_API_KEY');
let geminiIdx  = 0;

function getGeminiKey() {
  if (!geminiKeys.length) return null;
  const k = geminiKeys[geminiIdx % geminiKeys.length];
  geminiIdx = (geminiIdx + 1) % geminiKeys.length;
  return k;
}

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: true, limit: '25mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const apiLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 20,
  message: { error: 'Too many requests. Please wait a moment.' },
});

// ─── ML-Disguised Error Messages ─────────────────────────────────────────────
const ML_ERRORS = {
  no_key:     'The recognition model is not initialized. Please contact the administrator.',
  no_image:   'No image data received by the recognition engine. Please try again.',
  rate_limit: 'The recognition engine is at capacity. Please wait a moment and try again.',
  overloaded: 'The recognition engine is currently busy. Please try again shortly.',
  auth:       'Model authentication failed. Please contact the administrator.',
  generic:    'The model encountered an unexpected issue. Please try with a clearer image.',
};

// ─── System Prompt ────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are an expert handwriting recognition system specialized in reading ALL types of handwritten content. Analyze the provided image with extreme care.

CRITICAL INSTRUCTIONS:
1. Transcribe ALL visible handwritten or printed text EXACTLY as written
2. PRESERVE the original line-by-line structure — if text appears on multiple lines, output it on multiple lines
3. Recognize EVERY individual digit, letter, word, and number — even if they are written separately, close together, or in a grid/column layout
4. For multi-digit numbers (like 123, 4567, 98765), read them as complete numbers, not individual digits
5. For multiple separate numbers or words on different lines, transcribe EACH line on its own line
6. If the image contains mathematical equations or expressions, write them in plain text notation (e.g., 2+3=5)
7. If the image contains drawings or diagrams mixed with text, describe the drawings AND transcribe the text
8. Do NOT skip any content — even faint, small, or partially visible text should be transcribed
9. Rate your confidence level honestly based on legibility

Examples of expected output:
- Single digit "7" → output: 7
- Multiple digits "1 2 3" → output: 1 2 3
- Multi-digit number "456" → output: 456  
- Multiple lines:
  "Hello
   World
   123" → output:
  Hello
  World
  123

Respond EXACTLY in this format (keep the delimiters):
---RECOGNIZED_TEXT---
[Put the full transcribed text here preserving all line breaks, or a description if it's a drawing]
---CONFIDENCE---
[High/Medium/Low]
---ANALYSIS---
[Brief note: language detected, writing style, number of lines detected, legibility, any special symbols]`;

// ─── Gemini Vision via REST API (no SDK needed) ───────────────────────────────
async function callGemini(imageBase64, mimeType, userPrompt) {
  const key = getGeminiKey();
  if (!key) throw { code: 'no_key' };

  // Confirmed available models from the API (in order of preference)
  const models = [
    'gemini-2.0-flash',
    'gemini-2.5-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash-lite',
  ];

  let lastError = null;
  for (const model of models) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;

    // Force JPEG mime type for canvas images (Gemini works better with jpeg)
    const safeMime = (mimeType === 'image/png' || mimeType === 'image/jpeg') ? mimeType : 'image/jpeg';

    const body = {
      contents: [{
        parts: [
          {
            inline_data: {
              mime_type: safeMime,
              data: imageBase64,
            },
          },
          { text: SYSTEM_PROMPT + '\n\n' + (userPrompt || 'Recognize and transcribe ALL handwriting in this image. Include every digit, number, word, and line. Preserve the multi-line structure exactly as written.') },
        ],
      }],
      generationConfig: {
        temperature: 0.1,
        maxOutputTokens: 4096,
      },
    };

    let r, json;
    try {
      // Add timeout with AbortController (30 seconds)
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30000);
      
      r = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(timeout);
      json = await r.json();
    } catch (netErr) {
      console.log(`[InkMind] Network error on ${model}: ${netErr.message}`);
      // If it's an abort/timeout, try next model
      if (netErr.name === 'AbortError') {
        console.log(`[InkMind] Request to ${model} timed out, trying next...`);
      }
      continue;
    }

    if (r.ok) {
      const text = json?.candidates?.[0]?.content?.parts?.[0]?.text || '';
      if (text) {
        console.log(`[InkMind] ✅ Success with ${model}`);
        return text;
      }
      // Empty response — blocked by safety or empty image
      const reason = json?.candidates?.[0]?.finishReason || 'UNKNOWN';
      console.log(`[InkMind] ${model} returned empty (reason: ${reason})`);
      if (reason === 'SAFETY') continue; // try next model
      return 'The model could not extract text from this image. Please try a clearer image.';
    }

    const errCode = r.status;
    const errMsg  = json?.error?.message || json?.error?.status || r.statusText || '';
    console.log(`[InkMind] ${model} => HTTP ${errCode}: ${errMsg}`);

    // Auth errors are fatal — no point trying other models with same key
    if (errCode === 401 || errCode === 403) throw { code: 'auth' };
    
    // For rate limits and overload, try the NEXT model before giving up
    if (errCode === 429 || errCode === 503 || errCode === 502) {
      console.log(`[InkMind] ${model} is rate-limited/overloaded, trying next model...`);
      lastError = errCode === 429 ? 'rate_limit' : 'overloaded';
      continue;
    }
    // 404 = model not available for this key tier → try next
    // 400 = bad request → try next model
    continue;
  }

  throw { code: lastError || 'generic' };
}

// ─── Parse structured response ────────────────────────────────────────────────
function parseResponse(content) {
  let recognizedText = content.trim();
  let confidence = 'Medium';
  let analysis = '';

  const textMatch = content.match(/---RECOGNIZED_TEXT---\s*([\s\S]*?)\s*---CONFIDENCE---/);
  if (textMatch) recognizedText = textMatch[1].trim();

  const confMatch = content.match(/---CONFIDENCE---\s*([\s\S]*?)\s*---ANALYSIS---/);
  if (confMatch) confidence = confMatch[1].trim();

  const analysisMatch = content.match(/---ANALYSIS---\s*([\s\S]*)/);
  if (analysisMatch) analysis = analysisMatch[1].trim();

  return { recognizedText, confidence, analysis };
}

// ─── Extract base64 and mimeType from data URI ────────────────────────────────
function parseDataURI(dataUri) {
  const match = dataUri.match(/^data:([^;]+);base64,(.+)$/);
  if (match) return { mimeType: match[1], base64: match[2] };
  return { mimeType: 'image/png', base64: dataUri };
}

// ─── /api/recognize ───────────────────────────────────────────────────────────
app.post('/api/recognize', apiLimiter, async (req, res) => {
  try {
    const { image, prompt } = req.body;
    if (!image) return res.status(400).json({ error: ML_ERRORS.no_image });
    if (!geminiKeys.length) return res.status(500).json({ error: ML_ERRORS.no_key });

    const { mimeType, base64 } = parseDataURI(image);
    const userPrompt = prompt || 'Please recognize and transcribe all handwriting visible in this image.';

    const content = await callGemini(base64, mimeType, userPrompt);
    const result = parseResponse(content);

    return res.json({
      success: true,
      ...result,
      engine: 'InkMind Recognition Engine v2',
    });

  } catch (err) {
    const code = err?.code || 'generic';
    return res.status(500).json({ error: ML_ERRORS[code] || ML_ERRORS.generic });
  }
});

// ─── /api/health ──────────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    engine: 'InkMind Recognition Engine v2',
    keysLoaded: geminiKeys.length,
    timestamp: new Date().toISOString(),
  });
});

// ─── /api/test ────────────────────────────────────────────────────────────────
app.get('/api/test', (req, res) => {
  res.json({
    geminiKeysLoaded: geminiKeys.length,
    geminiKeyPrefix:  geminiKeys[0] ? geminiKeys[0].substring(0, 10) + '...' : 'none',
    status: geminiKeys.length > 0 ? 'ready' : 'no_key_configured',
  });
});

// ─── SPA fallback ─────────────────────────────────────────────────────────────
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ─── Startup self-test ───────────────────────────────────────────────────────
async function selfTest(key) {
  const tiny = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==';
  const model = 'gemini-2.0-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [
          { inline_data: { mime_type: 'image/png', data: tiny } },
          { text: 'What color is this image? One word.' }
        ]}],
        generationConfig: { maxOutputTokens: 10 }
      })
    });
    const json = await r.json();
    if (r.ok) {
      const reply = json?.candidates?.[0]?.content?.parts?.[0]?.text || 'ok';
      console.log(`✅ Gemini API working! Test reply: "${reply.trim()}"`);
    } else {
      const msg = json?.error?.message || json?.error?.status || r.statusText;
      console.error(`❌ Gemini API error (${r.status}): ${msg}`);
      console.error('   → Check your GEMINI_API_KEY in .env or Vercel environment variables');
    }
  } catch (e) {
    console.error('❌ Network error reaching Gemini:', e.message);
  }
}

// ─── Export for Vercel (serverless) + local start ─────────────────────────────
module.exports = app;

// Only start the HTTP server when running locally (not on Vercel)
if (process.env.NODE_ENV !== 'production' || process.env.LOCAL_DEV === 'true') {
  app.listen(PORT, async () => {
    console.log(`\n🚀 InkMind Recognition Server`);
    console.log(`   http://localhost:${PORT}`);
    console.log(`\n🔑 Gemini keys loaded: ${geminiKeys.length}`);
    if (!geminiKeys.length) {
      console.warn('\n⚠️  Add GEMINI_API_KEY to your .env file!');
      console.warn('   Get free key at: https://aistudio.google.com/app/apikey\n');
    } else {
      console.log('🧪 Running self-test...');
      await selfTest(geminiKeys[0]);
      console.log('');
    }
  });
}

