import torch
from langchain.schema import HumanMessage, SystemMessage
import torch
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import ValidationError
from utils import Chatbot
import yaml
from ml_model import get_inference


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32
config_path = './config.yaml'

ui_dir = os.path.dirname(os.getcwd())
# ui_dir = '../'

with open(config_path, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
        message = data['system_message']
        openai_api_key = data['openai_api_key']
        audio_path = os.path.join(ui_dir, data['audio_path'])

    except yaml.YAMLError as exc:
        print(exc)



# Check if the directory of the file exists, and if not, create it
directory = os.path.dirname(audio_path)
if not os.path.exists(directory):
    os.makedirs(directory)

sys_massage = SystemMessage(content=message)
chatbot = Chatbot(device=device, openai_api_key=openai_api_key, audio_path=audio_path)
chatbot.conversation_buffer.add_message(sys_massage)


app = Flask(__name__)
CORS(app)



@app.route("/chat", methods=['GET'])
def get_text_and_audio():
    global c
    user_message = request.args.get('user_message', '').strip()
    if user_message == '' :
        chatbot.conversation_buffer.reset_buffer()
        chatbot.conversation_buffer.add_message(sys_massage)
        response = chatbot.predict()
        return jsonify(response)

    elif user_message != '':

        try:
            ## TODO
            # extract the correct choice of the user from his response
            user_response = HumanMessage(content=user_message)
            chatbot.conversation_buffer.add_message(user_response)
            response = chatbot.predict()
            if 'answers_dict' in response['data']['text']:
                data = response['data']['text']
                result = get_inference(data)
                response['data']['text'] = result
                chatbot.save_audio(result)
                response['data']['audio'] = f'/static/audio/{chatbot.audio_id}.wav'
            return jsonify(response)

        except ValidationError as e:
            return jsonify({'error': 'Invalid user message', 'details': str(e)}), 400

    else:
        return jsonify({'error': 'No user message provided'}), 400
    
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5050")
    
