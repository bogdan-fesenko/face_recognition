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

import cv2
import dlib
import time
import random


gpu_memory_fraction = 0.2
facenet_model_checkpoint = os.path.dirname(__file__) + "/recognition/model_checkpoints/20180402-114759/"
classifier_model = os.path.dirname(__file__) + "/classification/model_checkpoints/my_classifier_1_128d.pkl"
debug = False


class Face:
    def __init__(self):
        self.id = None
        self.name = None
        self.bounding_box = None
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
        self.faces = [] # here is stored all current information about faces on the frame

        self.max_observations_before_recognize = 5  # so we choose 1 of 5 best frontal-view face and recognize it
        self.max_archived_n_frames = 3  # for tracker: number of frames to keep track archived with same info, then delete


    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)


        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        detected_faces = self.detect.find_faces(image) # list of objects Face()
        self.faces = self.tracker.match_tracks(detected_faces, self.faces, self.max_archived_n_frames)

        # gather faces we need to recognize for now
        faces_for_recognition = []
        faces_not_for_recognition = []
        for face in self.faces:
            if (face.n_observations >= self.max_observations_before_recognize or face.not_updated_n_frames > self.max_archived_n_frames) and face.id < 0:
                faces_for_recognition.append(face)
            else:
                faces_not_for_recognition.append(face)


        # Generate encodings and classify new faces
        if len(faces_for_recognition) > 0 :
            time_now = time.time()
            all_encodings = self.encoder.generate_embedding_dlib_128(faces_for_recognition, image) # inference on all cropped faces from frame
            # all_encodings = self.encoder.generate_embedding(faces)
            print(len(faces_for_recognition),":",round(1000*(time.time() - time_now),0)) # time purely wasted on face encoding

            # classify all new faces
            new_recognized_faces = []
            for i, face in enumerate(faces_for_recognition):
                if debug:
                    cv2.imshow("Face: " + str(i), face.image)

                face.embedding = np.squeeze(np.array(all_encodings[i]))
                face = self.identifier.identify(face)
                new_recognized_faces.append(face)

        self.faces = faces_not_for_recognition + new_recognized_faces
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
                updated_faces.append(face)

        faces = updated_faces



        # iterate over faces and match with new detected faces on current frame
        updated_faces = []
        for i, face in enumerate(faces):
            # get face from 'faces' with highest iou with current detected face. (faces - general class of current faces under tracking)
            best_match_new_face = max(detected_faces, key=lambda x_detected_faces: iou(face.bounding_box, x_detected_faces.bounding_box))

            # if IOU above threshold we connect faces as same and update tracking information
            if iou(face.bounding_box, best_match_new_face.bounding_box) >= self.sigma_iou:
                # update face
                face.bounding_box = best_match_new_face.bounding_box
                face.n_observations += 1
                face.not_updated_n_frames = 0

                # laso update image if detection_confidence better
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
            face.color = "%03x" % random.randint(0, 0xFFF) # get new random color
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

        self.db = None


    def identify(self, face):
        threshold_proba_recognized = 0.6

        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            max_proba = round(np.amax(predictions), 2)
            if max_proba > threshold_proba_recognized:
                face.id = self.add_unknown_face(face)

                return 'unknown', max_proba
            else:
                return self.class_names[best_class_indices[0]], max_proba


        return face

    def add_unknown_face(self, face):



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


class Encoder:
    def __init__(self):
        # self.sess = tf.Session()
        # with self.sess.as_default():
        #     facenet.load_model(facenet_model_checkpoint) # init: facenet 512D model


        # init : dlib.face_recognition model (128D) FaceNet (ResNet_v1)
        face_rec_model_path = 'encoding/dlib_face_recognition_resnet_model_v1.dat'
        face_align_model_path = 'encoding/shape_predictor_5_face_landmarks.dat'

        self.dlib_facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        self.dlib_predictor = dlib.shape_predictor(face_align_model_path)

    def generate_embedding(self, faces):
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
            face_shapes = self.dlib_predictor(face.image, bb_rectangle)
            bb_shapes.append(face_shapes)
            dlib_shapes.append(face_shapes)

        time_now = time.time()
        face_embeddings = self.dlib_facerec.compute_face_descriptor(frame, dlib_shapes)
        print("only face embedding:",len(faces), ":", round(1000 * (time.time() - time_now), 0))
        return face_embeddings






class Detection:
    # face detection parameters
    # minsize = 20  # minimum size of face
    # threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    # factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        # class for working with mobilenet_ssd net
        self.Detector = detection.detect_face_ssd.Detector()
        self.sess, self.detection_graph = self.Detector.create_model()

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

            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        # face only has objects : detection_confidence, container_image, bounding_box, image
        return faces
