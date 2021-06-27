#!/usr/bin/env python3
from flask import Flask, render_template, request,session,redirect,url_for
from chatbot import ask, append_interaction_to_chat_log
import string
import random
import os
import argparse
import io
import time
import numpy as np


from PIL import Image
import tensorflow as tf

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  
  
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]



labels = load_labels("labels.txt")

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

  
# initializing size of string 
N = 10
  
res = ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k = N))

app = Flask(__name__)
secret_key = res
app.config['SECRET_KEY'] = res
app.static_folder = 'static'

res_string = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get",methods=["POST", "GET"])
def get_bot_response():
        global res_string
        if request.method == "GET":
          
          
            if res_string is not None:
              
                print("the string is available",res_string)
                incoming_msg = str(res_string)
                chat_log = session.get('chat_log')
                print(chat_log)

                answer = ask(incoming_msg, chat_log)
                session['chat_log'] = append_interaction_to_chat_log(incoming_msg, answer,
                                                                     chat_log)
                print("res string",res_string)
                res_string =None
           
                return str(answer)
            userText = request.args.get('msg')
            userid = request.args.get('userid')
            incoming_msg = userText
            chat_log = session.get('chat_log')
            print(chat_log)

            answer = ask(incoming_msg, chat_log)
            session['chat_log'] = append_interaction_to_chat_log(incoming_msg, answer,
                                                                 chat_log)
            print("res string",res_string)
           

            return str(answer)
        if request.method == "POST":
                

            file = request.files['file']
            print("File", file)
                  # retrieve file from html file-picker
            upload = request.files.getlist("file")[0]
            print("File name: {}".format(upload.filename))
            filename = upload.filename

            print("file name:", filename)

              # file support verification
            ext = os.path.splitext(filename)[1]
            print("extension name:",ext)
            if (ext == ".jpg" or ext==".jpeg" or ext ==".png" or ext == ".JPEG"):
                print("File accepted")
              
                file.save("input.jpg")
                image = Image.open("input.jpg").convert('RGB').resize((width, height),
                                                  Image.ANTIALIAS)
                start_time = time.time()
                results = classify_image(interpreter, image)
                elapsed_ms = (time.time() - start_time) * 1000
                label_id, prob = results[0]

                annotate_text = '%s %.2f\n%.1fms' % (labels[label_id], prob,
                                                            elapsed_ms)
                result = labels[label_id]
                print(result)
                res_string = result
                #print(annotate_text)
                return redirect(url_for('home'))
            else:
                return render_template("error.html", message="The selected file is not supported"), 400
          
           

app.run(host='localhost',debug=True, port=8080)




 



