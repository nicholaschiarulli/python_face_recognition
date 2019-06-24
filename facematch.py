import face_recognition

image_of_bill = face_recognition.load_image_file('./img/known/Bill Gates.jpg')


bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

unknown_image = face_recognition.load_image_file(
    './img/unknown/d-trump.jpg')


unkown_face_encoding = face_recognition.face_encodings(unknown_image)[0]


# comapre faces

results = face_recognition.compare_faces(
    [bill_face_encoding], unkown_face_encoding)

if results[0]:
    print('This is Bill Gates')
else:
    print('this is not bill gates')
