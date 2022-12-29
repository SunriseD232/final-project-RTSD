import requests
from flask import Flask, jsonify, request
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras import models, layers, optimizers, callbacks, applications, backend, utils
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
from PIL import Image
app = Flask(__name__)


def nn():
    
    #loading custom model 
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\SunriseD\\Desktop\\project_work\\yolov5\\runs\\train\\exp6\\weights\\best.pt', force_reload=True)

    img = "C:\\Users\\SunriseD\\Desktop\\project_work\\road_signs\\test_image\\test_img.png"

    results = model(img)
    results.print()

    #getting bb
    results.xyxy[0].tolist() #where 1-4 pos - xy cor, 5 pos - accuracy, 6 pos - class

    coor = []
    img = cv2.imread ( 'C:\\Users\\SunriseD\\Desktop\\project_work\\road_signs\\test_image\\test_img.png')
    for i in range (len(results.xyxy[0].tolist())):
      x_min = int(results.xyxy[0].tolist()[i][0])
      y_min = int(results.xyxy[0].tolist()[i][1])
      w = int(results.xyxy[0].tolist()[i][2])
      h = int(results.xyxy[0].tolist()[i][3])
      confidence = str(results.xyxy[0].tolist()[i][4])
      string = ' confidence:' + confidence + ' bb: {x_min} ' + str(x_min) + ' {y_min} ' + str(y_min)+ ' {x_max} '  + str(w)+ ' {y_max} '  + str(h)

      coor.append (string)
      roi_img = img[y_min: h, x_min: w]

      cv2.imwrite(f"C:/Users/SunriseD/Desktop/project_work/test img/data_for_datagen/Cropped_Image{i}.jpg", roi_img)


    #Константы
    IMAGE_SHAPE = (128, 128)
    BATCH_SIZE = 100
    NUM_CLASSES = 50 #Число предсказуемых классов

    def create_model(transfer_model):
      model = models.Sequential([
            transfer_model,
            layers.Dense(512, activation='elu', kernel_regularizer=regularizers.L1L2(l2=0.001, l1=0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='elu', kernel_regularizer=regularizers.L1L2(l2=0.001, l1=0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.L1L2(l2=0.001, l1=0.005)) 
        ])
      return model

    baseModel = applications.MobileNetV2(weights="imagenet", include_top=False,
    input_shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3), pooling='avg')

    model = create_model(baseModel)

    model.load_weights("C:/Users/SunriseD/Desktop/project_work/Model/best_model.h5")
    train_datagen = image.ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        'C:/Users/SunriseD/Desktop/project_work/test img',
        target_size=IMAGE_SHAPE)
    pred = model.predict_generator(train_generator)

    target = ['1_11 - Опасный поворот',
              '1_11_1 - Опасный поворот',
              '1_16 - Неровная дорога',
              '1_17 - Искусственная неровность',
              '1_2 - Железнодорожный переезд без шлагбаума',
              '1_20_2 - Сужение дороги',
              '1_22 - Пешеходный переход',
              '1_23 - Дети',
              '1_25 - Дорожные работы',
              '1_8 - Светофорное регулирование',
              '2_1 - Главная дорога',
              '2_2 - Конец главной дороги',
              '2_3 - Пересечение со второстепенной дорогой',
              '2_3_2 - Примыкание второстепенной дороги',
              '2_3_3 - Примыкание второстепенной дороги',
              '2_4 - Уступите дорогу',
              '2_5 - Движение без остановки запрещено',
              '3_1 - Въезд запрещен',
              '3_13 - Ограничение высоты',
              '3_18 - Поворот запрещен',
              '3_20 - Обгон запрещен',
              '3_21 - Конец зоны запрещения обгона', 
              '3_24 - Ограничение максимальной скорости',
              '3_27 - Остановка запрещена',
              '3_28 - Стоянка запрещена',
              '3_4_1 - Движение грузовых автомобилей запрещено',

              '4_1_1 - Движение прямо',
              '4_1_2 - Движение направо(налево) ',
              '4_1_4 - Движение прямо или направо(налево)',
              '4_2_1 - Объезд препятствия',
              '4_2_2 - Объезд препятствия слева',
              '4_2_3 - Объезд препятствия справа или слева',
              '5_14 - Полоса для маршрутных транспортных средств',
              '5_15_1 - Направления движения по 2-м полосам',
              '5_15_2 - Направления движения по полосам',
              '5_15_2_2 - Направления движения по полосам', 
              '5_15_3 - Начало полосы',
              '5_15_5 - Конец полосы',
              '5_15_7 - Направление движения по полосам',
              '5_16 - Место остановки автобуса и (или) троллейбуса',
              '5_19_1 - Пешеходный переход',
              '5_20 - Искусственная неровность',
              '5_5 - Дорога с односторонним движением',
              '5_6 - Конец дороги с односторонним движением',
              '6_16 - Стоп-линия',
              '6_4 - Место стоянки',
              '6_6 - Подземный пешеходный переход',
              '7_3 - Автозаправочная станция',
              '7_5 - Мойка автомобилей',
              '8_13 - Направление главной дороги'
              ]
    all_result = {}
    len_all = []
    predict = np.argmax(pred, axis =1)
    for i in predict:
      len_all.append (target[i])
    for i in range(len(len_all)):
        all_result[len_all[i]] = coor[i]
    return all_result


@app.route("/detect", methods=['GET', 'POST'])
def connect():
    with open ('C:/Users/SunriseD/Desktop/project_work/road_signs/test_image/test_img.png','wb') as pol:
        pol.write(request.data)
    all_result = nn()
    print (all_result)
    return jsonify(all_result)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
