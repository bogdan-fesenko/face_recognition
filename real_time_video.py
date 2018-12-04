# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
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
import argparse
import sys
import time
import numpy as np

import cv2

import face
import cProfile



def add_overlays(frame, faces, frame_rate, use_normalized_coordinates=False):
    [im_height, im_width] = np.asarray(frame.shape)[0:2]

    if faces is not None:
        for face in faces:

            face_bb = face.bounding_box

            # draw box
            if use_normalized_coordinates:
                (face_bb[0], face_bb[1], face_bb[2], face_bb[3]) = (face_bb[0] * im_width, face_bb[1] * im_height,
                                                                    face_bb[2] * im_width, face_bb[3] * im_height)
            face_bb = face_bb.astype(int)

            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)

            # draw face name and confidence
            if face.name is not None:
                cv2.putText(frame, face.name+'. conf:'+str(face.proba) , (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    # draw FPS and inference time
    cv2.putText(frame, str(round(frame_rate,1)) + " fps. inf time: " + str(int(1000/frame_rate)) + 'ms' , (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def main(args):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 1  # seconds
    frame_rate = 0.0001
    frame_count = 0

    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture('test_face-detection_barca_fullhd.mp4')
    # video_capture.set(3, 1920)
    # video_capture.set(4, 1080)


    face_recognition = face.Recognition()
    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        face.debug = True


    # init models
    # detector, encoder, classifier = init_models()
    all_frame_count = 0
    while True:
        probs = []
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        all_frame_count += 1
        if all_frame_count > 10*30:
            if (frame_count % frame_interval) == 0:
                faces = face_recognition.identify(frame)

                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = round(frame_count / (end_time - start_time), 5)
                    if frame_rate == 0:
                        frame_rate = 0.00001
                    start_time = time.time()
                    frame_count = 0

            # Visualize bounding boxes
            add_overlays(frame, faces, frame_rate/3)#, use_normalized_coordinates=True)

            frame_count += 1
            cv2.imshow('Video', frame)
        if all_frame_count > 60*30:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    # cProfile.run('main(parse_arguments(sys.argv[1:]))')
    main(parse_arguments(sys.argv[1:]))
