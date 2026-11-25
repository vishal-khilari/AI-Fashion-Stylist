import unittest
from unittest.mock import MagicMock, patch
import json
from app import app
import gemini_helper


class TestChatbotIntentRules(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch('gemini_helper.get_gemini_client')
    def test_greeting_hi(self, mock_get_client):
        """Rule 1: If someone says 'Hi', respond with a simple greeting (type: text)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "text",
            "reply": "Hello there! Your personal stylist is here and ready to help. What can I style for you today? ✨"
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = (mock_client, None)

        resp = self.client.post('/chat_api', json={'message': 'Hi'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('type'), 'text')
        self.assertIn('Hello', data.get('reply', ''))
        self.assertIsNone(data.get('data'), "Should not generate 3 options for greetings")

    @patch('gemini_helper.get_gemini_client')
    def test_how_are_you(self, mock_get_client):
        """Rule 2: If someone asks 'How are you?', respond naturally (type: text)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "text",
            "reply": "I'm doing wonderfully, thank you for asking! Ready to help you discover the perfect outfit. What's on your agenda? ✨"
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = (mock_client, None)

        resp = self.client.post('/chat_api', json={'message': 'How are you?'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('type'), 'text')
        self.assertTrue(len(data.get('reply', '')) > 0)
        self.assertIsNone(data.get('data'))

    @patch('gemini_helper.get_gemini_client')
    def test_general_fashion_question(self, mock_get_client):
        """Rule 3: If someone asks a general fashion question, answer directly (type: text)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "text",
            "reply": "Chinos are made from lightweight cotton twill and are versatile for smart-casual wear, while dress trousers are typically tailored from wool or blended fabrics with sharp creases for formal settings."
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = (mock_client, None)

        resp = self.client.post('/chat_api', json={'message': 'What is the difference between chinos and trousers?'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('type'), 'text')
        self.assertTrue('Chinos' in data.get('reply', ''))
        self.assertIsNone(data.get('data'))

    @patch('gemini_helper.get_gemini_client')
    def test_outfit_recommendations_three_options(self, mock_get_client):
        """Rule 4: If someone asks for outfit recommendations, provide 3 complete options (type: suggestions)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "suggestions",
            "reply": "Here are 3 complete looks for you: ✨",
            "suggestions": [
                {
                    "outfit": "Midnight blue velvet tuxedo with black satin peak lapels",
                    "footwear": "Patent leather oxford shoes",
                    "grooming": "Clean taper fade and groomed stubble",
                    "accessories": "Silk bow tie, mother-of-pearl cufflinks, dress watch"
                },
                {
                    "outfit": "Classic black three-piece suit with crisp white dress shirt",
                    "footwear": "Polished black cap-toe derbies",
                    "grooming": "Neatly combed side part",
                    "accessories": "Silver tie clip and white silk pocket square"
                },
                {
                    "outfit": "Tailored charcoal double-breasted suit with subtle pinstripe",
                    "footwear": "Burgundy monk strap leather shoes",
                    "grooming": "Polished matte pomade hairstyle",
                    "accessories": "Burgundy knit tie and patterned pocket square"
                }
            ]
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = (mock_client, None)

        resp = self.client.post('/chat_api', json={'message': 'What should I wear for a formal evening charity gala?'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('type'), 'suggestions')
        suggestions = data.get('data', [])
        self.assertEqual(len(suggestions), 3, "Must provide exactly 3 complete options for outfit requests")
        for s in suggestions:
            self.assertIn('outfit', s)
            self.assertIn('footwear', s)
            self.assertIn('grooming', s)
            self.assertIn('accessories', s)

    @patch('gemini_helper.get_gemini_client')
    def test_three_outfit_ideas_explicit(self, mock_get_client):
        """Rule 5: If someone asks for 3 outfit ideas, provide 3 options (type: suggestions)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "suggestions",
            "reply": "Here are 3 complete looks for you: ✨",
            "suggestions": [
                {"outfit": "Linen shirt and tailored shorts", "footwear": "Boat shoes", "grooming": "Textured crop", "accessories": "Tortoiseshell sunglasses"},
                {"outfit": "Camp-collar shirt and chinos", "footwear": "Espadrilles", "grooming": "Natural wave", "accessories": "Woven leather bracelet"},
                {"outfit": "Light knit polo and white jeans", "footwear": "Clean white low-top sneakers", "grooming": "Slicked back", "accessories": "Minimalist watch"}
            ]
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = (mock_client, None)

        resp = self.client.post('/chat_api', json={'message': 'Give me 3 outfit ideas for a summer rooftop party'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('type'), 'suggestions')
        self.assertEqual(len(data.get('data', [])), 3)

    @patch('gemini_helper.get_gemini_client')
    def test_follow_up_on_existing_outfit(self, mock_get_client):
        """Rule 6: If someone follows up on an existing outfit, answer that specific follow-up directly (type: text)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "text",
            "reply": "Yes, pairing crisp white minimalist sneakers with Option 1 will give it a fresh, elevated smart-casual vibe. Just ensure the sneakers are clean and unblemished!"
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = (mock_client, None)

        history = [
            {'role': 'user', 'text': 'What should I wear to a beach party?'},
            {'role': 'assistant', 'text': 'Option 1: Linen button-down and tailored shorts with boat shoes. Option 2: Tropical shirt with espadrilles.'}
        ]
        resp = self.client.post('/chat_api', json={
            'message': 'Can I wear white leather sneakers with Option 1 instead?',
            'history': history
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('type'), 'text', "Follow-ups must NOT generate three new options")
        self.assertIn('sneakers', data.get('reply', '').lower())
        self.assertIsNone(data.get('data'))

    def test_missing_message_error(self):
        """Ensure missing message is handled with a 400 error."""
        resp = self.client.post('/chat_api', json={})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('error', data)


if __name__ == '__main__':
    unittest.main()
