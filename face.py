# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import detection.detect_face_ssd
import facenet
import database

import cv2
import dlib
import time
import random

# import insightface tf model
from encoding.tfinsightface import model, configs
from encoding.MXNETinsightface import model
from encoding.kerasopenface import model#, fr_utils, inception_blocks_v2
# installed mxnet-cu90 via PIP

# Aligner imports
import alignment.frontalize as frontalize
import alignment.facial_feature_detector as feature_detection
import alignment.camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
# import alignment.check_resources as check
import matplotlib.pyplot as plt



gpu_memory_fraction = 0.2
facenet_model_checkpoint = os.path.dirname(__file__) + "/recognition/model_checkpoints/20180402-114759/"
classifier_model = os.path.dirname(__file__) + "/classification/model_checkpoints/my_classifier_1_128d.pkl"
debug = False


class Face:
    def __init__(self):
        self.id = None
        self.name = None
        self.bounding_box = None
        self.prev_bounding_box = None
        self.image = None
        self.detection_confidence = None
        self.container_image = None
        self.embedding = None
        self.identification_proba = None
        self.n_observations = 1
        self.not_updated_n_frames = 0 # iterator for : number of frames we keep face track archived, after delete. Also if
        self.color = None



class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()
        self.tracker = Tracker()
        self.aligner = Aligner()
        self.faces = [] # here is stored all current information about faces on the frame

        self.max_observations_before_recognize = 15  # so we choose 1 of 5 best frontal-view face and recognize it
        self.max_archived_n_frames = 20  # for tracker: number of frames to keep track archived with same info, then delete

        print("\nclass Recognition was succesfully initializated\n")
    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)


        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        # detect faces on whole frame
        detected_faces = self.detect.find_faces(image) # list of objects Face()
        # track faces and match with previous frame
        self.faces = self.tracker.match_tracks(detected_faces, self.faces, self.max_archived_n_frames)


        # select faces we need to recognize now
        faces_for_recognition = []
        faces_not_for_recognition = []
        for face in self.faces:
            if (face.n_observations >= self.max_observations_before_recognize or face.not_updated_n_frames > self.max_archived_n_frames) and face.id < 0:
                faces_for_recognition.append(face)
            else:
                faces_not_for_recognition.append(face)


        self.faces = faces_not_for_recognition
        # Generate encodings and classify new faces
        if len(faces_for_recognition) > 0 :
            time_now = time.time()

            # align faces (frontalize)
            faces_for_recognition = self.aligner.align_faces(faces_for_recognition)

            # RUN INFERENCE ON FACE ENCODER
            # all_encodings = self.encoder.generate_embedding_dlib_128(faces_for_recognition, image) # inference on all cropped faces from frame
            all_encodings = self.encoder.generate_embeddings_common(faces_for_recognition)

            # all_encodings = self.encoder.generate_embedding(faces)
            print(len(faces_for_recognition),":",round(1000*(time.time() - time_now),0)) # time purely wasted on face encoding

            # classify all new faces
            new_recognized_faces = []
            for i, face in enumerate(faces_for_recognition):
                if debug:
                    cv2.imshow("Face: " + str(i), face.image)

                face.embedding = np.squeeze(np.array(all_encodings[i]))
                # here we get face as output and this is identified known/unknown face from DB
                face = self.identifier.identify(face)
                new_recognized_faces.append(face)

            self.faces += new_recognized_faces
        return self.faces


class Tracker:
    def __init__(self):

        self.sigma_l = 0
        self.sigma_h = 0.9  # high detection threshold
        self.sigma_iou = 0.5
        self.t_min = 2

        self.tracks = []

    def match_tracks(self, detected_faces, faces, max_archived_n_frames):

        # update faces info and delete old tracks
        updated_faces = []
        for face in faces:
            if not face.not_updated_n_frames > max_archived_n_frames:
                face.not_updated_n_frames += 1

                prev_bounding_box = face.prev_bounding_box
                face.prev_bounding_box = face.bounding_box
                if prev_bounding_box is not None:
                    face.bounding_box = [bb + (bb - prev_bb) for bb,prev_bb in zip(face.bounding_box, prev_bounding_box)]
                    # for i in range(4):
                    #     face.bounding_box[i] = face.bounding_box[i] + (face.bounding_box[i] - prev_bounding_box[i])

                updated_faces.append(face)

        faces = updated_faces


        # print("faces",faces)
        # print("detected faces:", detected_faces)
        if len(detected_faces) > 0:

            if len(faces) > 0:
                # iterate over faces and match with new detected faces on current frame
                updated_faces = []
                for i, face in enumerate(faces):
                    # check if we didn't delete all detected faces
                    if len(detected_faces) > 0:
                        # get face from 'faces' with highest iou with current detected face. (faces - general class of current faces under tracking)
                        best_match_new_face = max(detected_faces, key=lambda x_detected_faces: self.iou(face.bounding_box, x_detected_faces.bounding_box))

                        # if IOU above threshold we connect faces as same and update tracking information
                        if self.iou(face.bounding_box, best_match_new_face.bounding_box) >= self.sigma_iou:
                            # update face
                            face.bounding_box = best_match_new_face.bounding_box
                            face.n_observations += 1
                            face.not_updated_n_frames = 0
                            face.container_image = best_match_new_face.container_image

                            # also update image if detection_confidence better
                            if best_match_new_face.detection_confidence > face.detection_confidence:
                                face.detection_confidence = best_match_new_face.detection_confidence
                                face.image = best_match_new_face.image

                            # updated_faces.append(face)

                            # remove from best matching detection from detections
                            del detected_faces[detected_faces.index(best_match_new_face)]

                        else:
                            # if face isn't matched with any new detected face
                            if face.id >= 0:
                                # we lost recognized face
                                pass
                            if face.id < 0:
                                # we lost unrecognized face
                                pass

                        updated_faces.append(face)

            # create new track with detected faces that wasn't matched with any existing track
            new_faces = []
            for face in detected_faces:
                face.id = -1 - len(updated_faces) # id starts from -1 and -> -100
                face.color = ",".join([str(random.randint(0, 255)) for i in range(3)])  # get new random color. Ex: "1,123,15"
                new_faces.append(face)

            faces = updated_faces + new_faces

        return faces


    def iou(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.
        Args:
            bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
            bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        Returns:
            int: intersection-over-onion of bbox1, bbox2
        """

        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]

        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2

        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return size_intersection / size_union

class Identifier:
    def __init__(self):
        # with open(classifier_model, 'rb') as infile:
        #     self.model, self.class_names = pickle.load(infile)

        # setup connection to a database

        self.db = database.DatabaseWorker()

    def identify(self, face):
        threshold_proba_recognized = 0.6

        if face.embedding is not None:
            face = self.db.get_distances(face)

            #
            #
            #
            #
            # predictions = self.model.predict_proba([face.embedding])
            # best_class_indices = np.argmax(predictions, axis=1)
            # max_proba = round(np.amax(predictions), 2)
            # if max_proba > threshold_proba_recognized:
            #     face.id = self.add_unknown_face(face)
            #
            #     return 'unknown', max_proba
            # else:
            #     return self.class_names[best_class_indices[0]], max_proba


        return face

    # def identify(self, face):
    #     threshold_proba_recognized = 1
    #
    #     if face.embedding is not None:
    #         predictions = self.model.predict_proba([face.embedding])
    #         best_class_indices = np.argmax(predictions, axis=1)
    #         max_proba = round(np.amax(predictions), 2)
    #         if max_proba < threshold_proba_recognized:
    #             return 'unknown', max_proba
    #         else:
    #             return self.class_names[best_class_indices[0]], max_proba


class Aligner:
    def __init__(self):
        this_path = os.path.dirname(os.path.abspath(__file__))+'/alignment'
        # check for dlib saved weights for face landmark detection
        # if it fails, dowload and extract it manually from
        # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
        # check.check_dlib_landmark_weights()
        # load detections performed by dlib library on 3D model and Reference Image
        self.model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')
        # load query image
        # img = cv2.imread("test.jpg", 1)
        # plt.title('Query Image')
        # plt.imshow(img[:, :, ::-1])
        # # extract landmarks from the query image
        # # list containing a 2D array with points (x, y) for each face detected in the query image
        # lmarks = feature_detection.get_landmarks(img)
        # plt.figure()
        # plt.title('Landmarks Detected')
        # plt.imshow(img[:, :, ::-1])
        # plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1])

        # # perform camera calibration according to the first face detected
        # proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])

        # load mask to exclude eyes from symmetry
        self.eyemask = np.asarray(io.loadmat(this_path+'/frontalization_models/eyemask.mat')['eyemask'])

        # load dlib's face landmarks predictor
        predictor_path = this_path + "/dlib_models/shape_predictor_68_face_landmarks.dat"
        self.dlib_shape_predictor = dlib.shape_predictor(predictor_path)




    def _shape_to_np(self, shape):
        np_shape = []
        for i in range(68):
            np_shape.append((shape.part(i).x, shape.part(i).y,))
        np_shape = np.asarray(np_shape, dtype='float32')
        return np_shape

    def align_faces(self, faces):

        img = faces[0].image

        # face_bb = faces[0].bounding_box
        # bb_rectangle = dlib.rectangle(face_bb[0],face_bb[1],face_bb[2],face_bb[3])
        bb_rectangle = dlib.rectangle(0, 0, img.shape[0], img.shape[1])
        lmark = self.dlib_shape_predictor(img, bb_rectangle)
        lmark = self._shape_to_np(lmark)
        # print("type and len lmarks:", type(lmark), len(lmark), lmark)

        # lmarks = feature_detection.get_landmarks(img)
        # lmark = lmarks[0]

        plt.figure()
        plt.title('Landmarks Detected')
        plt.imshow(img[:, :, ::-1])
        plt.scatter(lmark[:, 0], lmark[:, 1])
        plt.show()



        # perform camera calibration according to the first face detected
        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(self.model3D, lmark)
        # print("proj_matrix type and value:", type(proj_matrix), proj_matrix)


        for face in faces:
            # perform frontalization
            frontal_raw, frontal_sym = frontalize.frontalize(face.image, proj_matrix, self.model3D.ref_U, self.eyemask)

            frontal_raw = frontal_raw[:, :, ::-1]
            frontal_sym = frontal_sym[:, :, ::-1]


            # plt.figure()
            # plt.title('Frontalized no symmetry')
            # plt.imshow(frontal_raw)
            # plt.figure()
            # plt.title('Frontalized with soft symmetry')
            # plt.imshow(frontal_sym)
            # plt.show()
            #
            # plt.figure()
            # plt.title('Frontalized no symmetry')
            # plt.imshow(frontal_raw[:, :, ::-1])
            # plt.figure()
            # plt.title('Frontalized with soft symmetry')
            # plt.imshow(frontal_sym[:, :, ::-1])
            # plt.show()

            face.image = frontal_raw[160-80:160+80, 160-80:160+80,:]
            # cv2.imshow('img', face.image)
            # face.image = frontal_raw

        return faces


class Encoder:
    def __init__(self):
        print("\nmsg: Start loading face encoding model...\n")
        # ----------------- init : facenet (davidsandberg model)
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint) # init: facenet 512D model


        # # -----------------init : dlib.face_recognition model (128D) FaceNet (ResNet_v1)
        # face_rec_model_path = 'encoding/dlib_face_recognition_resnet_model_v1.dat'
        # face_align_model_path = 'encoding/shape_predictor_5_face_landmarks.dat'
        # self.dlib_facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        # self.dlib_shape_predictor = dlib.shape_predictor(face_align_model_path)


        # # --------------- init : tf-insightface model (resnet_v1_50)
        # self.insightface_model = model.BaseServer(model_fp=configs.face_describer_model_fp,
        #                                          input_tensor_names=configs.face_describer_input_tensor_names,
        #                                          output_tensor_names=configs.face_describer_output_tensor_names,
        #                                          device=configs.face_describer_device)

        # # ------------------init : insightface model (resnet_100E-IR_) (oficial mxnet implementation)
        # self.insightface_model = model.FaceModel()



        # self.openface_nn4_smallv7 = model.FaceModel()





        # self.encoder_model = self.openface_nn4_smallv7
        print("\nmsg: Finish loading face encoding model!\n")

    def generate_embeddings_common(self, faces):
        # all_encodings = self.generate_embedding_insightface_mxnet(faces)
        all_encodings = self.generate_embedding_facenet(faces)
        return all_encodings



    def generate_embedding_facenet(self, faces):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_faces = []
        for face in faces:
            prewhiten_faces.append(facenet.prewhiten(face.image))

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: prewhiten_faces, phase_train_placeholder: False}  # prewhiten faces - list. if 1 image : [prewhiten_face]
        time_now = time
        embeddings = self.sess.run(embeddings, feed_dict=feed_dict)
        # print("embeddings shape", embeddings.shape)

        return embeddings

    def generate_embedding_dlib_128(self, faces, frame):
        # Generate 128 embeddings using dlib's face recognition model (ResNet_v1)
        prewhiten_faces = []
        for face in faces:
            prewhiten_faces.append(facenet.prewhiten(face.image))

        face_embeddings = []
        bb_shapes = []
        dlib_shapes = dlib.full_object_detections()
        for face in faces:
            # print("face shape",prewhiten_faces[i].shape)
            # print(self.dlib_facerec.compute_face_descriptor(face, 160))
            face_bb = face.bounding_box
            bb_rectangle = dlib.rectangle(face_bb[0],face_bb[1],face_bb[2],face_bb[3])
            face_shapes = self.dlib_shape_predictor(face.container_image, bb_rectangle)
            # bb_shapes.append(face_shapes)
            dlib_shapes.append(face_shapes)

        time_now = time.time()
        face_embeddings = self.dlib_facerec.compute_face_descriptor(frame, dlib_shapes)
        print("only face embedding:",len(faces), ":", int(1000 * (time.time() - time_now)),'ms')
        return face_embeddings

    def generate_embedding_insightface(self, faces):  # tf-insightface omplementation (TF)

        input_data = []
        dropout_rate = 0.5
        # images = np.empty(shape=(112,112,3))

        # for face in faces[]:
        #     input_data = [face.image, dropout_rate]
            # images = np.stack((images, face.image), axis=0)

        images = np.stack( tuple([face.image for face in faces]), axis=0 )
        print("images shape", images.shape)
        input_data = [images, dropout_rate]
        all_embeddings = self.insightface_model.inference(data=input_data)
        print("all embeddings . len:", len(all_embeddings[0]))
        print("all embeddings shape . ", np.array(all_embeddings).shape)

        all_embeddings = np.array(all_embeddings[0])
        return all_embeddings

    def generate_embedding_insightface_mxnet(self, faces):  # oficial insightface implementation (mxNet)

        all_embeddings = []

        for face in faces:
            all_embeddings.append(self.insightface_model.get_feature(face.image))

        #
        # images = np.stack( tuple([face.image for face in faces]), axis=0 )
        # print("images shape", images.shape)
        # input_data = [images, dropout_rate]
        # all_embeddings = self.insightface_model.inference(data=input_data)
        # print("all embeddings . len:", len(all_embeddings[0]))
        # print("all embeddings shape . ", np.array(all_embeddings).shape)
        #
        # all_embeddings = np.array(all_embeddings[0])
        return all_embeddings

    def generate_embedding_openface_keras(self, faces):  # oficial insightface implementation (mxNet)

        all_embeddings = []

        for face in faces:
            all_embeddings.append(self.model.img_to_encoding(face.image))

        return all_embeddings






class Detection:
    # face detection parameters
    # minsize = 20  # minimum size of face
    # threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    # factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=10):#32): crop_size was 160
        # class for working with mobilenet_ssd net
        print("\nStart loading face-detection model...\n")
        self.Detector = detection.detect_face_ssd.Detector()
        self.sess, self.detection_graph = self.Detector.create_model()
        print("\nFinish loading face-detection model!\n")

        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.threshold = 0.7  # threshold for face detection

    def find_faces(self, image):
        faces = []

        # bounding_boxes = self.Detector.detect_face(image, self.sess, self.detection_graph)
        boxes = []
        scores = []
        classes = []
        num_detections = 0


        (boxes_all, scores_all, classes_all, num_detections_all) = self.Detector.detect_face(image, self.sess, self.detection_graph)
        img_size = np.asarray(image.shape)[0:2]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        for i in range(int(num_detections_all)):
            if scores_all[i] >= self.threshold:
                import time
                ymin, xmin, ymax, xmax = boxes_all[i]
                box = [xmin,ymin,xmax,ymax]
                boxes.append(box)

                scores.append(scores_all[i])
                num_detections += 1




        # when shape=1 np.squeeze func outputs a scalar value, but we need always array
        if len(boxes) <= 1:
            bounding_boxes = np.array(boxes)
        else:
            # bounding_boxes = np.squeeze(boxes)
            bounding_boxes = np.array(boxes)


        for bb, score in zip(bounding_boxes, scores):

            face = Face()
            # unnormalize coordinates!
            # print("BOX unnorm!:", bb)
            bb[0], bb[1], bb[2], bb[3] = bb[0]*img_size[1], bb[1]*img_size[0], bb[2]*img_size[1], bb[3]*img_size[0]
            # print("BOX norm!:", bb)

            face.detection_confidence = score
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1]-1)
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0]-1)

            if len(image.shape) == 3:
                cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            else:
                cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2]]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        # face only has objects : detection_confidence, container_image, bounding_box, image
        return faces
