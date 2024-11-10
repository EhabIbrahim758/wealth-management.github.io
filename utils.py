from langchain.chat_models import ChatOpenAI
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
import os 

class Chatbot:
    
    def __init__(self, device, openai_api_key, audio_path):
        self.device = device
        self.openai_api_key = openai_api_key
        self.var = ''
        self.audio_path = audio_path
        self.init_models()
        self.conversation_buffer = ConversationBuffer(max_tokens=5000)
        self.audio_id = 0
        
    def init_models(self):
        self.chat_model = ChatOpenAI(model_name="gpt-4-1106-preview",temperature=0.7, openai_api_key=self.openai_api_key)
        self.tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(self.device)
        self.tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    
    def split_prompt(self, response):

        # Split the prompt into two parts based on the 'choices :' identifier
        self.var += response
        if "Choices:" in response:
            parts = response.split("Choices:")
        else:
            return response, ""
        # Remove 'question :' from the first part and strip whitespace
        context_and_question = parts[0].replace("Question: ", "").strip()
        context_and_question = context_and_question.replace("Question:", "").strip()
        # Strip whitespace from the choices part
        choices = parts[1].strip()
        # Split the choices into a list
        choice_list = choices.split('\n')
        # Add <br> tag before each choice if not already present
        updated_choice_list = []
        for choice in choice_list:
            if not choice.startswith("<br>"):
                choice = "<br> " + choice
            updated_choice_list.append(choice)

        # Join the updated choice list back into a string
        updated_choices = '\n'.join(updated_choice_list)
        return context_and_question , updated_choices
    
    
    def save_audio(self, context_and_question):
        inputs = self.tts_tokenizer(context_and_question, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.tts_model(**inputs).waveform

        output = output.cpu()
        rate = self.tts_model.config.sampling_rate
        sf.write(os.path.join(self.audio_path, f'{self.audio_id}.wav'), output[0], rate)
    
    def predict(self):
        ai_response = self.chat_model(self.conversation_buffer.get_conversation())
        self.conversation_buffer.add_message(ai_response)

        response = ai_response.content
        context_and_question, choices = self.split_prompt(response)
        self.save_audio(context_and_question)
        
        response = {
            "status_code": 200,
            "status": "success",
            "data": {
                "text": "\n\n".join([context_and_question, choices]),
                "audio": f'/static/audio/{self.audio_id}.wav'
            }
        }
        self.audio_id += 1
        return response



class ConversationBuffer:
    def __init__(self, max_tokens=1000):
        self.max_tokens = max_tokens
        self.messages = []

    def reset_buffer(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)
        self._trim_buffer()

    def _trim_buffer(self):
        current_tokens = sum(len(message.content.split()) for message in self.messages)
        while current_tokens > self.max_tokens:
            # Remove the oldest messages until under max_tokens
            removed_message = self.messages.pop(0)
            current_tokens -= len(removed_message.content.split())

    def get_conversation(self):
        return self.messages


