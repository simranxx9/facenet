from scipy.spatial.distance import cosine
import mtcnn
from keras.models import load_model
from utils import *
import os
import json
import requests

def recognize(img,
              detector,
              encoder,
              encoding_dict,
              recognition_t=0.5,
              confidence_t=0.99,
              required_size=(160, 160), ):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'


        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)

            # print(name , " name")
            URL = "https://fdmlops.herokuapp.com/post"


            headers = {
                'Content-Type': 'application/json'
            }

            data = json.dumps({
                "firstname": name,
                "lastname": "na",
                "mobile_number": "8888888888",
                "description": "hi",
                "location":"japan"
            })
            response = requests.post(URL, data=data, headers=headers)
            print(response)

            data = json.loads(response.text)
            print(data)
            break

    return img


if __name__ == '__main__':
    def joiner(file_name):
        paths = os.path.dirname(os.path.abspath(__file__))
        paths = os.path.join(paths, file_name)
        return paths

    try:
        model = load_model(joiner('data/model/facenet_keras.h5'))
    except:
        print("error thrown")

    encoder_model = load_model('data/model/facenet_keras.h5')
    encodings_path = 'data/encodings/encodings.pkl'

    face_detector = mtcnn.MTCNN()
    face_encoder = encoder_model
    encoding_dict = load_pickle(encodings_path)

    vc = cv2.VideoCapture(0)
    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            print("no frame:(")
            break
        frame = recognize(frame, face_detector, face_encoder, encoding_dict)
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
