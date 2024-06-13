import json
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
import random
import urllib
import numpy as np
import cv2
from keras.models import load_model
import requests
from dotenv import load_dotenv
import os



def ansewer_for_message(event, api):

    msg = api.messages.getById(message_ids = event.message_id)
    photo_url = msg['items'][0]['attachments'][0]['photo']['sizes'][4]['url']
    print(photo_url)
    req = urllib.request.urlopen(photo_url)
    image = np.asarray(bytearray(req.read()) , dtype= np.uint8)
    image = cv2.imdecode(image, -1)
    image = cv2.resize(image, (624, 852), interpolation=cv2.INTER_AREA)
    faces = face_cascade.detectMultiScale(image)
    images =[]
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            croped = image[y:y + h, x:x + w]
            croped = cv2.resize(croped, (48, 48), interpolation=cv2.INTER_AREA)
            images.append(croped)

        images = np.array(images)
        classes = model.predict_classes(images)
        index = []
        for i, clas in enumerate(classes):
            if clas == 0:
                index.append(i)
        for (x, y, w, h) in np.array(faces)[index]:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv2.imwrite('photo.jpg', image)
    data = api.photos.getMessagesUploadServer(user_id= event.user_id)
    upload_url = data["upload_url"]
    files = {'photo': open("photo.jpg", 'rb')}

    response = requests.post(upload_url, files=files)
    result = json.loads(response.text)


    uploadResult = api.photos.saveMessagesPhoto(server=result["server"],
                                                  photo=result["photo"],
                                                  hash=result["hash"])

    att = 'photo{}_{}'.format(uploadResult[0]["owner_id"], uploadResult[0]["id"])
    api.messages.send(user_id= event.user_id,
                        message="randomTextMessage",
                        random_id= random.randint(1,100000),
                        attachment= att)


try:
    load_dotenv()
    vk_token = os.environ['vk_token']

    vk_session = vk_api.VkApi(token=vk_token)
    vk_ap = vk_session.get_api()

    model = load_model('test9902.hdf5')
    face_cascade = cv2.CascadeClassifier('cascade.xml')

    longpoll = VkLongPoll(vk_session)
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            ansewer_for_message(event, vk_ap)

except:
    print('error')




