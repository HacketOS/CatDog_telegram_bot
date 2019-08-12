import telebot
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf

def load_image(path, size = [150,150]):
  img = load_img(path, target_size = size)
  return img_to_array(img).reshape(1,150,150,3)

catdog_classifier = load_model('CatDog.h5', compile= False)
graph = tf.get_default_graph()

_token = '831937104:AAEdZQq6_CrpnkIVmteJPHYzBl6nkRl2AwM'

bot = telebot.TeleBot(_token)
@bot.message_handler(commands=['start'])
def start_message(message):
  bot.send_message(message.chat.id, 'Чё каво чепушило')

@bot.message_handler(content_types=['photo'])
def send_message(message):
    raw = message.photo[-1].file_id
    name = raw + ".jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(name,'wb') as new_file:
        new_file.write(downloaded_file)
    img = load_image(name)
    global graph
    with graph.as_default():
      answer = catdog_classifier.predict(img)[0]
    f_answer = 'cat: %.3f \ndog: %.3f' % tuple(answer)
    bot.send_message(message.chat.id, f_answer)
bot.polling()