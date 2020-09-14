import os
import cv2
import dlib
import shutil
import time
import onnx
import joblib
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import onnxruntime as ort
from imutils import face_utils
from onnx_tf.backend import prepare
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.core.files.storage import FileSystemStorage
passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
from onnx import optimizer


count = 0
frame_count = 0 
current_directory = os.getcwd()

#################################################################################################################################


def home(request):
    try:
        if request.method == 'POST' and request.FILES['myfile']:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            detected_video = detect(uploaded_file_url)
            return render(request, 'home.html', {
                "video_url": '/' + detected_video
            })

        if request.method == 'POST' and request.FILES['myfile2']:
            myfile2 = request.GET['myfile2']
            fs2 = FileSystemStorage()
            filename2 = fs2.save(myfile2.name, myfile2)
            uploaded_file_url2 = fs.url(filename2)
            detected_video2 = detect(uploaded_file_url2)
            return render(request, 'home.html', {
                "value": '/' + uploaded_file_url2
            })

        return render(request, 'home.html')
    except Exception as e:
        print(e)
        return render(request, 'home.html', {
            'no_file': True
        })



def clear(request):
    return render(request, 'home.html')


#################################################################################################################################


def area_of(left_top, right_bottom):
    pass
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    pass
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k =-1, candidate_size = 200):
    pass
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


#################################################################################################################################


def predict1(width, height, confidences, boxes, prob_threshold, iou_threshold=0.6, top_k=-1):
    pass
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

#################################################################################################################################
#################################################################################################################################
#                                    todo-> 
#                                        a.) augment(rotate) face frame
#                                        b.) only save best faces(manual deletion not required)
                             #           c.)  extracted faces interface (for deletion purposes)
#################################################################################################################################
###############################################################################################################################
def extract(uploaded_file_url2):
    from .models import userdata
    import datetime
    # pass
    global frame_count
    request=uploaded_file_url2
    file2 = request.FILES.get('myfile2')
    filedown2=default_storage.save(file2.name,file2)
    file2 = default_storage.open(filedown2)
    url2 = default_storage.url(filedown2) 
    n = request.POST['name']
    d1 = userdata(User = n, Video = url2, Date_joined = datetime.datetime.now())
    d1.save()
    # import pdb; pdb.set_trace()
    if os.path.exists(current_directory+'/face/extracted_images/{0}'.format(n)):
        print(os.path.exists(current_directory+'/face/extracted_images/{0}'.format(n)))
        return HttpResponse('<script> alert("name exists !"); window.history.go(-1); </script>')
    else :
        os.mkdir(current_directory+'/face/extracted_images/{0}'.format(n))
        path = os.path.join(os.getcwd(), '/face/extracted_images/{0}'.format(n))
        print(path,str(current_directory+'/'+url2))
        video_capture = cv2.VideoCapture(str(current_directory+'/'+url2))
        frame_count = 0
        while True:
            print(True)
            ret, raw_img = video_capture.read()
            # raw_img=cv2.flip(raw_img,-1)
            if frame_count % 5 == 0 and raw_img is not None:
                print('Inside if')
                # import pdb;pdb.set_trace()
                h, w, _ = raw_img.shape
                img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 480))
                img_mean = np.array([127, 127, 127])
                img = (img - img_mean) / 128
                img = np.transpose(img, [2, 0, 1])
                img = np.expand_dims(img, axis=0)
                img = img.astype(np.float32)

                # print('current_directory', current_directory)
                onnx_path = current_directory+"/face/model_train/ultra_light/ultra_light_models/ultra_light_640.onnx"
                # print('onnx_path',onnx_path)
                onnx_model = onnx.load(onnx_path)
                optimized_model = optimizer.optimize(onnx_model, passes)
                predictor = prepare(onnx_model)
                ort_session = ort.InferenceSession(onnx_path)
                input_name = ort_session.get_inputs()[0].name
                shape_predictor = dlib.shape_predictor(current_directory+'/face/model_train/facial_landmarks/shape_predictor_5_face_landmarks.dat')
                fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=105, desiredLeftEye=(0.2, 0.2))
                threshold = 0.60

                confidences, boxes = ort_session.run(None, {input_name: img})
                boxes, labels, probs = predict1(w, h, confidences, boxes, 0.7)
                if boxes.shape[0] > 0:
                    print('Inside boxes')
                    x1, y1, x2, y2 = boxes[0,:]
                    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                    aligned_face = fa.align(raw_img, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                    aligned_face = cv2.resize(aligned_face, (112,112))
                    cv2.imwrite(f'{path}/{frame_count}.jpg', aligned_face)
            frame_count += 1
            if frame_count == video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                break
        return HttpResponse('<script> alert("Faces extracted ! Please delete the faces "); window.history.go(-1); </script>')

##########################################################################################################

def train(i):
    # pass
    images = []
    names = []
    database_name=[]        
    IMAGES_BASE = os.path.join(os.getcwd(), 'face/extracted_images')
    # -------------------------------------------------------------------------------------------------------------------------------------
    dirs = os.listdir(IMAGES_BASE)
    #---------------------------------------------------------------------------------------trained name---------------|
    with open(str(current_directory+"/face/model_train/embeddings/embeddings.pkl"), "rb") as f:
        (new_embeddings1, new_names1) = pickle.load(f)
    trained_name=list(set(new_names1))
    print("trained_name=",trained_name)
    #--------------------------------------------------------------------------database name--------<
    for label in dirs:
      database_name.append(label)
    print("database_name=",database_name)

    uniq=set(database_name)-set(trained_name)
    uniq=list(uniq)
    print("new names(_unique_)",uniq)
    if len(uniq)==0:
        return HttpResponse('<script> alert("Model already trained successfully for given person, Click OK to go back"); window.history.go(-1); </script>')
 
    for label in dirs: 
      print("main loop")
      for name in uniq:
        print("name in uniq loop") 
        if name == label:
          print(f"Collecting {label}'s faces")
          for i, fn in enumerate(os.listdir(os.path.join(IMAGES_BASE, label))):
              img = cv2.imread(os.path.join(IMAGES_BASE,label,fn))
              img = img - 127.5
              img = img * 0.0078125
              images.append(img)
              names.append(label)
 #<<<<<<<<<<<<<<<<<<<<<<<<<<<deleting user face folder from extracted images(databases)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>             
          # shutil.rmtree(current_directory+'/extracted_images/{0}/'.format(label))   
 # ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo                 
          print("names=",set(names))
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print("loading checkpoint ...")
            saver = tf.train.import_meta_graph(str(current_directory+'/face/model_train/mfn/m1/mfn.ckpt.meta'))
            saver.restore(sess, str(current_directory+'/face/model_train/mfn/m1/mfn.ckpt'))
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")     
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            embeds = sess.run(embeddings, feed_dict=feed_dict)
            with open(str(current_directory+"/face/model_train/embeddings/new.pkl"),'wb') as f:
                pickle.dump((embeds, names), f)  
                print("created embedding for",set(names))
    #---------additon of two embeddings---------------------------------------------------------------------------------------------|
    print("adding-",set(names),"to",trained_name)
#----------------------------------------------------------------------------------loading new.pkl
    with open(str(current_directory+"/face/model_train/embeddings/new.pkl"), "rb") as f:      
        (new_embeddings1, new_names1) = pickle.load(f)
#----------------------------------------------loading old pickle
    file =open(str(current_directory+"/face/model_train/embeddings/embeddings.pkl"), "rb")
    old_data = pickle.load(file)                             
    new_embeddings = old_data[0]           
    new_names      = old_data[1]  
   
    final_embed= np.concatenate((new_embeddings,new_embeddings1))      #joning pickle emdedding
    final_namesk=new_names+new_names1                                    #joining pickle names
    with open(str(current_directory+"/face/model_train/embeddings/embeddings.pkl"),"wb") as f:
        pickle.dump((final_embed, final_namesk), f)                   #final joined embeddings.pkl
        print("final trained names =====>",set(final_namesk))
    # if set(final_namesk)-set(database_name)==None:
    return HttpResponse('<script> alert("Model trained successfully for given person, Click OK to go back"); window.history.go(-1); </script>')

#################################################################################################################################

def detectlive(r):
    # pass
    request = r
    camera_number = request.POST['dropdown']
    print('Getting live feed from camera {0}'.format(camera_number))
    global frame_count
    with open(str(current_directory+"/face/model_train/embeddings/embeddings.pkl"), "rb") as f:
        (saved_embeds, names) = joblib.load(f)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(str(current_directory+'/face/model_train/mfn/m1/mfn.ckpt.meta'))
            saver.restore(sess, str(current_directory+'/face/model_train/mfn/m1/mfn.ckpt'))
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            try:
                video_capture = cv2.VideoCapture('rtsp://ADMIN:1234@192.168.1.108:554/cam/realmonitor?channel={0}&subtype=0'.format(camera_number))
                # video_capture=cv2.VideoCapture('/home/relinns/Desktop/facenet/facenet/face_recognition/output/final_all3.mp4')
                while True:
                    ret, frame = video_capture.read()
                    frame=cv2.flip(frame,-1)
                    h, w, c = frame.shape
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (640, 480))
                    img_mean = np.array([127, 127, 127])
                    img = (img - img_mean) / 128
                    img = np.transpose(img, [2, 0, 1])
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32)

                    # print('current_directory', current_directory)
                    onnx_path = current_directory + '/face/model_train/ultra_light/ultra_light_models/ultra_light_640.onnx'
                    # print('onnx_path',onnx_path)
                    onnx_model = onnx.load(onnx_path)
                    predictor = prepare(onnx_model)
                    ort_session = ort.InferenceSession(onnx_path)
                    input_name = ort_session.get_inputs()[0].name
                    shape_predictor = dlib.shape_predictor(current_directory+'/face/model_train/facial_landmarks/shape_predictor_5_face_landmarks.dat')
                    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=105, desiredLeftEye=(0.2, 0.2))
                    threshold = 0.60

                    confidences, boxes = ort_session.run(None, {input_name: img})
                    boxes, labels, probs = predict1(w, h, confidences, boxes, 0.7)
                    faces = []
                    boxes[boxes<0] = 0
                    for i in range(boxes.shape[0]):
                        box = boxes[i, :]
                        x1, y1, x2, y2 = box
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        aligned_face = fa.align(frame, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                        aligned_face = cv2.resize(aligned_face, (112,112))
                        out=aligned_face
                        aligned_face = aligned_face - 127.5
                        aligned_face = aligned_face * 0.0078125
                        faces.append(aligned_face)
                    if len(faces)>0:
                        predictions = []
                        faces = np.array(faces)
                        feed_dict = { images_placeholder: faces, phase_train_placeholder:False }
                        embeds = sess.run(embeddings, feed_dict=feed_dict)
                        for embedding in embeds:
                            diff = np.subtract(saved_embeds, embedding)                   
                            dist = np.sum(np.square(diff), 1)
                            idx = np.argmin(dist)
                            if dist[idx] < threshold:
                                predictions.append(names[idx])
                                print(((str(dist[idx]*100)[:4])),'-'+names[idx]+'-'+'-'+str(count))
                                # cv2.imwrite(f'testoutput/{count}_{names[idx]}_{(str(dist[idx]*100)[:4])}.jpg', out)
                            else:
                                predictions.append("unknown")
                                print(((str(dist[idx]*100)[:4])),"-unknown"+'-'+'-'+str(count))
                        for i in range(boxes.shape[0]):
                            box = boxes[i, :]
                            text = f"{predictions[i]}"
                            x1, y1, x2, y2 = box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
                            cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, text, (x1 + 6, y2 - 9), font, 0.6, (255, 255, 255), 1)
                            # cv2.putText(frame, 'Press q to exit the live feed', (x1 + 10, y2 - 5), font, 0.4, (255, 255, 255), 1)
                    cv2.imshow("Streaming Camera {0}".format(camera_number),frame)
                    frame_count += 1
                    if frame_count == video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except:
                return HttpResponse('<script> alert("No camera found, Click OK to go back"); window.history.go(-1); </script>')
    video_capture.release()
    cv2.destroyAllWindows()    
    return HttpResponse('<script> window.history.go(-1); </script>')



### delete userdata from "embeddings.pkl" file  ##############################################################################################################################


def deleteuser(k):
    # pass
    from .models import userdata    
    IMAGES_BASE = os.path.join(os.getcwd(), 'face/extracted_images')
    request = k 
    user = request.POST['user']
    print(user)
    with open(str(current_directory+"/face/model_train/embeddings/embeddings.pkl"), "rb") as f:
        (saved_embeds, names) = joblib.load(f)
    if(user in names):
        print(" Deleting ",user,"from database")
        names_index = []
        for i in range(len(names)):
          if(names[i]==user):
             names_index.append(i)
        # print(names_index)
        start = names_index[0]
        end = names_index[-1]+1
        remaining_embeds = np.delete(saved_embeds, slice(start, end), 0)
        remaining_names = [s for s in names if s != user]
        with open(str(current_directory+"/face/model_train/embeddings/embeddings.pkl"),"wb") as f:
            pickle.dump((remaining_embeds, remaining_names), f)  
        # print(remaining_embeds.shape, type(remaining_embeds))
        # print(len(remaining_names), type(names))
        unique_users=set(remaining_names)
        print("Remained users for which our model is trained")
        print("=====================[Remained trained Users]========================")
        print(unique_users)
        
        userdata.objects.filter(User=user).delete()
        os.remove(str(current_directory+"/face/model_train/embeddings/new.pkl"))
        # if os.path.exists(current_directory+'/extracted_images/{0}'.format(user)):
    #         shutil.rmtree(current_directory+'/extracted_images/{0}/'.format(user))
    #         return HttpResponse('<script> alert("{0} deleted from Database and Model, Click OK to go back"); window.history.go(-1); </script>'.format(user))

    #     else:
    #         return HttpResponse('<script> alert("{0} deleted from Model, Click OK to go back"); window.history.go(-1); </script>'.format(user))
    # elif os.path.exists(current_directory+'/extracted_images/{0}'.format(user)):
    #     shutil.rmtree(current_directory+'/extracted_images/{0}/'.format(user))
        return HttpResponse('<script> alert("{0} deleted from Database, Click OK to go back"); window.history.go(-1); </script>'.format(user))
    else:
        return HttpResponse('<script> alert("No user named {0} exists in our Database and Model, PLease check username !"); window.history.go(-1); </script>'.format(user))
    # print((saved_embeds, names))




############### New two classes for -database file deletion - ##########################################################################################################################

# class SelectFileDelView(TemplateView):
#     """
#     This view is used to select a file from the list of files in the server.
#     After the selection, it will send the file to the server.
#     The server will then delete the file.
#     """
#     template_name = 'select_file_deletion.html'
#     parser_classes = FormParser
#     queryset = FileModel.objects.all()

#     def get_context_data(self, **kwargs):
#         """
#         This function is used to render the list of files in the MEDIA_ROOT in the html template
#         and to get the pk (primary key) of each file.
#         """
#         context = super().get_context_data(**kwargs)
#         media_path = settings.MEDIA_ROOT
#         myfiles = [f for f in listdir(media_path) if isfile(join(media_path, f))]
#         primary_key_list = []
#         for value in myfiles:
#             primary_key = FileModel.objects.filter(file=value).values_list('pk', flat=True)
#             primary_key_list.append(primary_key)
#         file_and_pk = zip(myfiles, primary_key_list)
#         context['filename'] = file_and_pk
#         return context

# #333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
# class FilesList(ListView):
#     """
#     ListView that display companies query list.
#     :param model: Specifies the objects of which model we are listing
#     :param template_name; Specifies the static display template file.
#     :param context_object_name: Custom defined context object value,
#                      this can override default context object value.
#     """
#     model = FileModel
#     template_name = 'files_list.html'
#     context_object_name = 'files_list'

##################################################################################################################
# class SelectPredFileView(TemplateView):
#     """
#     This view is used to select a file from the list of files in the server.
#     After the selection, it will send the file to the server.
#     The server will return the predictions.
#     """

#     template_name = 'select_file_predictions.html'
#     parser_classes = FormParser
#     queryset = FileModel.objects.all()

#     def get_context_data(self, **kwargs):
#         """
#         This function is used to render the list of files in the MEDIA_ROOT in the html template.
#         """
#         context = super().get_context_data(**kwargs)
#         media_path = settings.MEDIA_ROOT
#         myfiles = [f for f in listdir(media_path) if isfile(join(media_path, f))]
#         context['filename'] = myfiles
#         return context

############################################################################################################

def detect(uploaded_file_url):
    # pass
    global frame_count, count
    request=uploaded_file_url
    file = request.FILES.get('myfile')
    filedown=default_storage.save(file.name,file)
    file = default_storage.open(filedown)
    url = default_storage.url(filedown) 
    with open(str(current_directory+"/face/model_train/embeddings/embeddings.pkl"),"rb") as f:
        (saved_embeds, names) = joblib.load(f)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(str(current_directory+'/face/model_train/mfn/m1/mfn.ckpt.meta'))
            saver.restore(sess, str(current_directory+'/face/model_train/mfn/m1/mfn.ckpt'))
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            video_capture = cv2.VideoCapture(str(current_directory+'/'+url))
            if video_capture is None or not video_capture.isOpened():
                return HttpResponse('<script> alert("Warning: unable to open video source"); window.history.go(-1); </script>')
            else:
                writer = None
                while True:
                    ret, frame = video_capture.read()
                    h, w, c = frame.shape
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (640, 480))
                    img_mean = np.array([127, 127, 127])
                    img = (img - img_mean) / 128
                    img = np.transpose(img, [2, 0, 1])
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32)

                    # print('current_directory', current_directory)
                    onnx_path = current_directory+"/face/model_train/ultra_light/ultra_light_models/ultra_light_640.onnx"
                    # print('onnx_path',onnx_path)
                    onnx_model = onnx.load(onnx_path)
                    optimized_model = optimizer.optimize(onnx_model, passes)
                    predictor = prepare(onnx_model)
                    ort_session = ort.InferenceSession(onnx_path)
                    input_name = ort_session.get_inputs()[0].name
                    shape_predictor = dlib.shape_predictor(current_directory+'/face/model_train/facial_landmarks/shape_predictor_5_face_landmarks.dat')
                    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=105, desiredLeftEye=(0.2, 0.2))
                    threshold = 0.60

                    confidences, boxes = ort_session.run(None, {input_name: img})
                    boxes, labels, probs = predict1(w, h, confidences, boxes, 0.7)
                    faces = []
                    boxes[boxes<0] = 0
                    for i in range(boxes.shape[0]):
                        box = boxes[i, :]
                        x1, y1, x2, y2 = box
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        aligned_face = fa.align(frame, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                        aligned_face = cv2.resize(aligned_face, (112,112))
                        out=aligned_face
                        aligned_face = aligned_face - 127.5
                        aligned_face = aligned_face * 0.0078125
                        faces.append(aligned_face)
                    if len(faces)>0:
                        predictions = []
                        faces = np.array(faces)
                        feed_dict = { images_placeholder: faces, phase_train_placeholder:False }
                        embeds = sess.run(embeddings, feed_dict=feed_dict)
                        for embedding in embeds:
                            diff = np.subtract(saved_embeds, embedding)                   
                            dist = np.sum(np.square(diff), 1)
                            idx = np.argmin(dist)
                            if dist[idx] < threshold:
                                predictions.append(names[idx])
                                print(((str(dist[idx]*100)[:4])),'-'+names[idx]+'-'+'-'+str(count))
                                # cv2.imwrite(f'testoutput/{count}_{names[idx]}_{(str(dist[idx]*100)[:4])}.jpg', out)
                            else:
                                predictions.append("unknown")
                                print(((str(dist[idx]*100)[:4])),"-unknown"+'-'+'-'+str(count))
                            count+=1
                        for i in range(boxes.shape[0]):
                            box = boxes[i, :]
                            text = f"{predictions[i]}"
                            x1, y1, x2, y2 = box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
                            cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, text, (x1 + 6, y2 - 9), font, 0.6, (255, 255, 255), 1)
                        if writer is None:
                            # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                            new_url = current_directory+'/media/sample_data_output.avi'
                            print(new_url)
                            writer = cv2.VideoWriter(new_url, fourcc, 30,
                                                    (frame.shape[1], frame.shape[0]), True)
                        writer.write(frame)
                    cv2.imshow("Real Time Face Monitoring",frame)
                    frame_count += 1
                    if frame_count == video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        writer.release()        
        video_capture.release()
        cv2.destroyAllWindows() 
        print(new_url)
