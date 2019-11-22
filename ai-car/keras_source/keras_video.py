#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
from keras.models import model_from_json


def main():

    arg_parser = argparse.ArgumentParser(description='Execute keras model for recognition。')
    arg_parser.add_argument(
        '--model-file',
        required=True,
        help='model',
    )
    arg_parser.add_argument(
        '--weights-file',
        required=True,
        help='weight',
    )
    arg_parser.add_argument(
        '--video-type',
        choices=['file', 'camera'],
        default='camera',
        help='video type',
    )
    arg_parser.add_argument(
        '--source',
        default='/dev/video0',
        help='video source',
    )
    arg_parser.add_argument(
        '--input-width',
        type=int,
        default=48,
        help='video width',
    )
    arg_parser.add_argument(
        '--input-height',
        type=int,
        default=48,
        help='video height',
    )
    arg_parser.add_argument(
        '--gui',
        action='store_true',
        help='GUI',
    )

    args = arg_parser.parse_args()
    assert args.input_width > 0 and args.input_height > 0

    with open(args.model_file, 'r') as file_model:
        model_desc = file_model.read()
        model = model_from_json(model_desc)

    model.load_weights(args.weights_file)

    if args.video_type == 'file':  
        video_dev = cv2.VideoCapture(args.source)
        video_width = video_dev.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = video_dev.get(cv2.CAP_PROP_FRAME_HEIGHT)

    elif args.video_type == 'camera':  
        video_dev = cv2.VideoCapture(0)

    try:
        prev_timestamp = time.time()

        while True:
            ret, orig_image = video_dev.read()
            curr_time = time.localtime()

            
            if ret is None or orig_image is None:
                break

            
            resized_image = cv2.resize(
                orig_image,
                (args.input_width, args.input_height),
            ).astype(np.float32)
            normalized_image = resized_image / 255.0

     
            batch = normalized_image.reshape(1, args.input_height, args.input_width, 3)
            result_onehot = model.predict(batch)
            left_score, right_score, stop_score, other_score = result_onehot[0]
            class_id = np.argmax(result_onehot, axis=1)[0]

            if class_id == 0:
                class_str = 'left'
            elif class_id == 1:
                class_str = 'right'
            elif class_id == 2:
                class_str = 'stop'
            elif class_id == 3:
                class_str = 'other'

            recent_timestamp = time.time()
            period = recent_timestamp - prev_timestamp
            prev_timestamp = recent_timestamp

            print('Current Time:%02d:%02d:%02d ' % (curr_time.tm_hour, curr_time.tm_min, curr_time.tm_sec))
            print('Output:%.2f %.2f %.2f %.2f' % (left_score, right_score, stop_score, other_score))
            print('Type:%s' % class_str)
            print('Spent Time:%f' % period)
            print()

            
            if args.gui:
                cv2.imshow('', orig_image)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print('KeyboardInterrupt')


    video_dev.release()


if __name__ == '__main__':
    main()
