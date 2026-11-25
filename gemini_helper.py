"""
gemini_helper.py — Centralized Google Gen AI SDK Integration
-------------------------------------------------------------
Uses the official Google Gen AI SDK (google-genai) and gemini-2.5-flash.
Provides thread-safe access to Gemini API for:
  1. Fashion styling compliments/advice for single and multi-person outfits.
  2. Interactive AI stylist chatbot outfit recommendations (structured JSON).
"""

import os
import re
import json
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Default model as specified
MODEL_NAME = "gemini-2.5-flash"

_cached_client = None
_cached_key = None


def get_gemini_api_key() -> Optional[str]:
    """
    Safely load the Gemini API key.
    Priority:
      1. api_key.txt (local file)
      2. Environment variable: GEMINI_API_KEY
      3. Environment variable: GOOGLE_API_KEY
    Strips surrounding whitespace, quotes, and ignores placeholders.
    """
    candidate_key = None

    # 1. Check api_key.txt first
    key_file = "api_key.txt"
    if os.path.exists(key_file):
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Remove any accidental wrapping quotes (single or double)
                content = content.strip("\"'").strip()
                if content and not content.startswith("YOUR_") and len(content) >= 15:
                    candidate_key = content
        except Exception as e:
            logger.warning(f"Could not read {key_file}: {e}")

    # 2. Check environment variables if not found in api_key.txt
    if not candidate_key:
        for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            val = os.environ.get(env_var)
            if val:
                val = val.strip().strip("\"'").strip()
                if val and not val.startswith("YOUR_") and len(val) >= 15:
                    candidate_key = val
                    break

    return candidate_key


def get_gemini_client():
    """
    Get or create a google-genai Client instance.
    Returns (client, error_message).
    """
    global _cached_client, _cached_key

    api_key = get_gemini_api_key()
    if not api_key:
        return None, "Gemini API key not found. Please provide a valid API key in api_key.txt or set GEMINI_API_KEY."

    if _cached_client is not None and _cached_key == api_key:
        return _cached_client, None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        _cached_client = client
        _cached_key = api_key
        return client, None
    except ImportError:
        return None, "google-genai SDK is not installed. Please run: pip install google-genai"
    except Exception as e:
        logger.error(f"Failed to initialize Gemini GenAI client: {e}")
        return None, f"Client initialization failed: {str(e)}"


def generate_outfit_advice(persons_info: List[Dict[str, Any]]) -> str:
    """
    Generate styling feedback for detected person(s).
    persons_info is a list of dicts with:
      {'person_label': 'Person 1', 'predicted_class': 'Formal', 'confidence': 94.5}
    """
    client, err = get_gemini_client()
    if err or not client:
        return "You look great! (Add your Gemini API key in api_key.txt to unlock live AI Stylist compliments)."

    if not persons_info:
        return "Looking stylish! Step in front of the camera to see your outfit breakdown."

    try:
        if len(persons_info) == 1:
            p = persons_info[0]
            label = p.get('person_label', 'Your outfit')
            cls = p.get('predicted_class', 'attire')
            conf = p.get('confidence', 90.0)
            prompt = (
                f"Act as an expert high-fashion stylist. A person was detected wearing an outfit classified "
                f"as {cls} with {conf}% confidence. "
                "Give a short, energetic, and stylish 2-sentence fashion compliment and styling tip. "
                "Keep it punchy, positive, and hype them up!"
            )
        else:
            descriptions = []
            for p in persons_info:
                label = p.get('person_label', f"Person {p.get('person_id', 1)}")
                cls = p.get('predicted_class', 'attire')
                conf = p.get('confidence', 90.0)
                descriptions.append(f"{label}: wearing {cls} ({conf}% match)")
            desc_str = "; ".join(descriptions)
            prompt = (
                f"Act as an expert high-fashion stylist. A group of {len(persons_info)} people is in frame: "
                f"{desc_str}. "
                "Give a cohesive, witty, and energetic 2-sentence compliment on how their looks pair together "
                "or individual quick shout-outs. Be creative and hype up the whole group!"
            )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text.strip()

    except Exception as e:
        err_str = str(e)
        logger.warning(f"Gemini styling advice generation error: {err_str}")
        if "API_KEY_INVALID" in err_str or "400" in err_str:
            return "Gemini API key is invalid or expired. Please check your key in api_key.txt."
        return "You all look fantastic! (Stylist AI advice temporarily unavailable)."


def generate_chat_response(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Process chatbot message with conditional intent routing:
      1. Greetings & casual conversation ("Hi", "How are you?") -> Direct natural conversational response.
      2. General fashion & style questions -> Direct, concise answers.
      3. Follow-ups on an existing outfit -> Direct specific answers without generating 3 new options.
      4. Outfit recommendation requests or asking for 3 outfit ideas -> Exactly 3 complete outfit options
         with (outfit, footwear, grooming, accessories).

    Returns dict with keys: 'success', 'type' ('suggestions' or 'text'), and 'data' or 'reply' or 'error'.
    """
    if not user_message or not user_message.strip():
        return {'success': False, 'error': 'No message provided'}

    client, err = get_gemini_client()
    if err or not client:
        return {
            'success': True,
            'type': 'text',
            'reply': "Please set your Gemini API key in api_key.txt or the GEMINI_API_KEY environment variable to chat with the stylist!"
        }

    # Format history if provided so multi-turn follow-ups have context
    history_context = ""
    if history and isinstance(history, list):
        history_lines = []
        for h in history[-8:]:  # Keep recent context window
            role = "User" if h.get("role") == "user" else "Stylist"
            text = h.get("text") or h.get("content") or ""
            if text:
                history_lines.append(f"{role}: {text}")
        if history_lines:
            history_context = "Recent Conversation History:\n" + "\n".join(history_lines) + "\n\n"

    prompt = f"""You are StyleSense AI, an expert high-fashion stylist and personal fashion consultant.
Analyze the user's message and any conversation history, and determine the user's intent according to these strict rules:

1. GREETINGS & CASUAL CHAT (e.g., 'Hi', 'Hello', 'Hey', 'How are you?'):
   - If someone says 'Hi', respond with a simple, warm, stylish greeting.
   - If someone asks 'How are you?', respond naturally and politely like an enthusiastic personal stylist.
   - Output JSON format: {{"type": "text", "reply": "<your friendly response>"}}

2. GENERAL FASHION / STYLING QUESTIONS (e.g., 'What colors match with navy?', 'What is the difference between chinos and trousers?', 'Are loafers formal?', 'How should a suit jacket fit?'):
   - Answer the question directly, concisely, and informatively.
   - Output JSON format: {{"type": "text", "reply": "<your direct answer>"}}

3. FOLLOW-UPS TO AN EXISTING OUTFIT / RECOMMENDATION (e.g., 'Can I wear white sneakers with Option 1 instead?', 'What jacket would go with the second look?', 'Which of those three is best if it gets chilly?'):
   - Answer that specific follow-up directly. DO NOT generate three new options.
   - Output JSON format: {{"type": "text", "reply": "<your direct follow-up advice>"}}

4. OUTFIT RECOMMENDATION REQUESTS & 3 OUTFIT IDEAS (e.g., 'What should I wear to a beach wedding?', 'I need an outfit for a job interview', 'Give me 3 outfit ideas for a date', 'Can you style me for a dinner party?'):
   - Provide exactly 3 complete, distinct styling options.
   - Output JSON format: {{
       "type": "suggestions",
       "reply": "Here are 3 complete looks for you: ✨",
       "suggestions": [
         {{
           "outfit": "<description of main clothing/garments>",
           "footwear": "<suggested footwear/shoes>",
           "grooming": "<hair, facial grooming, or makeup>",
           "accessories": "<jewelry, watch, bag, belt, sunglasses, etc.>"
         }},
         {{
           "outfit": "...",
           "footwear": "...",
           "grooming": "...",
           "accessories": "..."
         }},
         {{
           "outfit": "...",
           "footwear": "...",
           "grooming": "...",
           "accessories": "..."
         }}
       ]
     }}

{history_context}Current User message: "{user_message.strip()}"

Respond in valid JSON only. Do not wrap in markdown code blocks."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        reply_text = response.text.strip() if response and response.text else ""

        # Remove markdown fences if present
        if reply_text.startswith("```json"):
            reply_text = reply_text[7:]
        elif reply_text.startswith("```"):
            reply_text = reply_text[3:]
        if reply_text.endswith("```"):
            reply_text = reply_text[:-3]
        reply_text = reply_text.strip()

        try:
            parsed = json.loads(reply_text)
            if isinstance(parsed, dict):
                resp_type = parsed.get('type', 'text')
                if resp_type == 'suggestions' and isinstance(parsed.get('suggestions'), list):
                    return {
                        'success': True,
                        'type': 'suggestions',
                        'data': parsed['suggestions'],
                        'reply': parsed.get('reply', 'Here are 3 complete looks for you: ✨')
                    }
                elif resp_type == 'text':
                    return {
                        'success': True,
                        'type': 'text',
                        'reply': parsed.get('reply', '')
                    }
            elif isinstance(parsed, list) and len(parsed) > 0:
                # Raw list of recommendations fallback
                return {'success': True, 'type': 'suggestions', 'data': parsed}
        except json.JSONDecodeError:
            pass

        return {'success': True, 'type': 'text', 'reply': reply_text or "Hello! I am your AI Stylist. How can I help style you today?"}

    except Exception as e:
        err_str = str(e)
        logger.warning(f"Failed to generate chat response: {err_str}")
        if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
            return {
                'success': False,
                'error': "Gemini API key is invalid. Please update api_key.txt with a valid API key."
            }
        elif "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            clean_msg = user_message.strip().lower()
            if clean_msg in ("hi", "hello", "hey", "hi there", "hello there", "good morning", "good evening"):
                return {
                    'success': True,
                    'type': 'text',
                    'reply': "Hello! I am StyleSense AI, your personal fashion stylist. ✨ Ask me what to wear for an event or for styling advice!"
                }
            elif "how are you" in clean_msg:
                return {
                    'success': True,
                    'type': 'text',
                    'reply': "I'm doing wonderful, thank you for asking! Ready to help you put together the perfect look. What's on your fashion radar today? ✨"
                }
            return {
                'success': False,
                'error': "Gemini API rate limit or free-tier daily quota exceeded (20 requests/day). Please wait for quota to refresh or supply a key in api_key.txt."
            }
        return {
            'success': False,
            'error': f"The stylist is unavailable right now. (Details: {err_str})"
        }
