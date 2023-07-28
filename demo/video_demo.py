# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector
from projects import *


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--batchsize', type=int, default=8, help='Batchsize of prediction')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument(
        '--vid-stride', default=1, type=int, help='Video frame-rate stride')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device)

    video_reader = cv2.VideoCapture(args.video)
    width = video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, fps,
            (int(width), int(height)))
        frame_batch = []

    num_frame = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in tqdm(range(int(num_frame))):
        video_reader.grab()  # .read() = .grab() followed by .retrieve()
        if args.show:
            ret, frame = video_reader.retrieve()
            if ret:
                result = inference_detector(model, frame)
                frame = model.show_result(frame, result, score_thr=args.score_thr)
                cv2.namedWindow('video', 0)
                mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            if i % args.vid_stride == 0:
                ret, frame = video_reader.retrieve()
                if ret:
                    frame_batch.append(frame)
                    if (len(frame_batch) == args.batchsize) | (i == (num_frame - 1)):
                        result = inference_detector(model, frame_batch)
                        for j in range(len(frame_batch)):
                            frame = model.show_result(
                                frame_batch[j], result[j], score_thr=args.score_thr)
                            video_writer.write(frame)
                        frame_batch = []

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
