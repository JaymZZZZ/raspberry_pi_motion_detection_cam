#!/usr/bin/python3
import argparse
import datetime
import logging
import os
import signal
import smtplib
import socket
import sys
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

import colorama
from PIL import Image
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput

from colorama import init as colorama_init, Back
from colorama import Fore
from colorama import Style

# setLevel(logging.WARNING) seems to have no impact
logging.getLogger("picamera2").disabled = True


def command_line_handler(signum, frame):
    res = input("Ctrl-C was pressed. Do you really want to exit? y/n ")
    if res == 'y':
        motion_detector.stop()


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Motion detection for Raspberry Pi Camera Module 2 with optional E-Mail send.')
    parser.add_argument('--preview', help='Enables the preview window', required=False, action='store_true')
    parser.add_argument('--preview-x', type=int, default=100,
                        help='Preview window location x-axis')
    parser.add_argument('--preview-y', type=int, default=200,
                        help='Preview window location y-axis')
    parser.add_argument('--preview-width', type=int, default=800,
                        help='Preview window width')
    parser.add_argument('--preview-height', type=int, default=600,
                        help='Preview window height')
    parser.add_argument('--zoom', type=float, default=1.0,
                        help='zoom factor (0.5 is half of the resolution and therefore the zoom is x 2)',
                        required=False)
    parser.add_argument('--width', type=int, default=1280, help='Camera resolution width for high resolution',
                        required=False)
    parser.add_argument('--height', type=int, default=720, help='Camera resolution height for high resolution',
                        required=False)
    parser.add_argument('--lores-width', type=int, default=320, help='Camera resolution width for low resolution',
                        required=False)
    parser.add_argument('--lores-height', type=int, default=240, help='Camera resolution height for low resolution',
                        required=False)
    parser.add_argument('--min-pixel-diff', type=float, default=7.2,
                        help='Minimum number of pixel changes to detect motion (determined with numpy by calculating the mean of the squared pixel difference between two frames)',
                        required=False)
    parser.add_argument('--capture-lores', help='enables capture of lores buffer', action='store_true')
    parser.add_argument('--recording-dir', default='./recordings/', help='directory to store recordings',
                        required=False)
    parser.add_argument('--delete-local-recordings',
                        help='Delete local recordings after email is sent',
                        required=False, action='store_true')
    parser.add_argument('--snapshot-only',
                        help='Sends an email with just an image snapshot and not the whole video',
                        required=False, action='store_true')
    parser.add_argument('--max-recording-length-seconds', type=int, default=0,
                        help='Limit recording length to seconds')
    parser.add_argument('--recipient', type=str, help='Email address to send the recordings to', required=False)
    parser.add_argument('--email-username', type=str, help='Email account username (from)', required=False)
    parser.add_argument('--email-password', type=str, help='Password of the email account to send the recordings',
                        required=False)
    parser.add_argument('--smtp-server', type=str, default='smtp.gmail.com', help='SMTP Server', required=False)
    parser.add_argument('--smtp-port', type=int, default=465, help='SMTP Port', required=False)
    parser.add_argument('--debug',
                        help='Enables debug mode',
                        required=False, action='store_true')

    return parser.parse_args()


class MotionDetector:
    """This class contains the main logic for motion detection."""
    __MAX_TIME_SINCE_LAST_MOTION_DETECTION_SECONDS = 5.0

    def __init__(self, args: argparse.Namespace):
        """MotionDetector

        :param args: command line arguments
        """
        self.__picam2 = None
        self.__encoder = None
        self.__encoding = False
        self.__start_time_of_last_recording = None
        self.__time_of_last_motion_detection = None
        self.__display_interval = 100
        self.__tick = 0

        self.__diff_history = []
        self.__diff_history_count = 1000
        self.__diff_min = 9999
        self.__diff_max = 0
        self.__diff_average = 0

        self.__zoom_factor = args.zoom
        self.__lores_width = args.lores_width
        self.__lores_height = args.lores_height
        self.__width = args.width
        self.__height = args.height
        self.__min_pixel_diff = args.min_pixel_diff
        self.__capture_lores = args.capture_lores

        self.__recording_dir = args.recording_dir
        self.__delete_local_recordings = args.delete_local_recordings
        self.__snapshot_only = args.snapshot_only
        self.__preview_x = args.preview_x
        self.__preview_y = args.preview_y
        self.__preview_width = args.preview_width
        self.__preview_height = args.preview_height
        self.__max_recording_length_seconds = args.max_recording_length_seconds

        self.__recipient = args.recipient
        self.__email_username = args.email_username
        self.__email_password = args.email_password
        self.__smtp_server = args.smtp_server
        self.__smtp_port = args.smtp_port

        self.__set_up_camera(args.preview)

        self.__debug_mode = args.debug

    def start(self):

        if self.__debug_mode:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        """
        Starts the camera and runs the loop.
        """
        self.__picam2.start()
        self.__picam2.start_encoder(self.__encoder)

        self.__set_zoom_factor()

        self.__loop()

    def __loop(self):
        """
        Runs the actual motion detection loop that, optionally, triggers sends the recording via email.
        """
        w, h = self.__lsize
        previous_frame = None

        while True:
            try:
                current_frame = self.__picam2.capture_buffer("lores" if self.__capture_lores else "main")
                current_frame = current_frame[:w * h].reshape(h, w)
                if previous_frame is not None:
                    hist_diff = self.__calculate_histogram_difference(current_frame, previous_frame)
                    self.store_diff_history(hist_diff)
                    if hist_diff > self.__min_pixel_diff and not self.__is_max_recording_length_exceeded() and not self.__encoding:
                        if not self.__encoding:
                            self.__start_time_of_last_recording = datetime.datetime.now()
                            self.log_info(f"Starting new recording: {self.__start_time_of_last_recording}")
                            self.__start_recording()
                        self.__time_of_last_motion_detection = datetime.datetime.now()
                        self.log_movement_start(f"Motion Detected - Diff: {hist_diff}")
                    elif self.__is_max_recording_length_exceeded():
                        self.log_info(
                            f"Max recording time exceeded after {(datetime.datetime.now() - self.__start_time_of_last_recording).total_seconds()} seconds")
                        self.__write_recording_to_file()
                    else:
                        if self.__is_max_time_since_last_motion_detection_exceeded():
                            self.log_info("Max time since last motion detection exceeded")
                            self.log_movement_end(f"Motion No-Longer Detected - Diff: {hist_diff}")
                            self.__write_recording_to_file()
                previous_frame = current_frame
            except Exception as e:
                self.log_error(f"An error occurred in the motion detection loop: {e}")
                continue

    def __calculate_histogram_difference(self, current_frame, previous_frame):
        current_image = Image.fromarray(current_frame)
        previous_image = Image.fromarray(previous_frame)

        current_hist = current_image.histogram()
        previous_hist = previous_image.histogram()

        hist_diff = sum([abs(c - p) for c, p in zip(current_hist, previous_hist)]) / len(current_hist)

        return hist_diff

    def __is_max_recording_length_exceeded(self):
        return self.__max_recording_length_seconds > 0 and self.__start_time_of_last_recording is not None and (
                (
                        datetime.datetime.now() - self.__start_time_of_last_recording).total_seconds() >= self.__max_recording_length_seconds
        )

    def __is_max_time_since_last_motion_detection_exceeded(self):
        return self.__encoding and self.__time_of_last_motion_detection is not None and \
            ((
                     datetime.datetime.now() - self.__time_of_last_motion_detection).total_seconds() > self.__MAX_TIME_SINCE_LAST_MOTION_DETECTION_SECONDS)

    def __start_recording(self):
        self.__encoder.output.fileoutput = self.__get_recording_file_path()
        self.__encoder.output.start()
        self.__encoding = True

    def __write_recording_to_file(self):
        self.__write_snapshot_to_file()
        file_path = self.__get_recording_file_path()
        snapshot_path = self.__get_snapshot_file_path()
        self.log_info(f"Writing file: {file_path}")
        self.__encoder.output.stop()
        _, file_name = os.path.split(file_path)
        if self.__snapshot_only:
            self.__upload_file(file_path=snapshot_path)
        else:
            self.__upload_file(file_path=file_path)
        self.__encoding = False
        self.__start_time_of_last_recording = None

    def __create_snapshot(self):
        request = self.__picam2.capture_request()
        request.save("main", self.__get_snapshot_file_path())
        request.release()

    def __write_snapshot_to_file(self):
        file_path = self.__get_snapshot_file_path()
        self.log_info(f"Writing snapshot file: {file_path}")
        self.__create_snapshot()

    def __get_recording_file_path(self):
        return f"{self.__recording_dir}{self.__start_time_of_last_recording.isoformat()}.h264"

    def __get_snapshot_file_path(self):
        return f"{self.__recording_dir}{self.__start_time_of_last_recording.isoformat()}.jpeg"

    def __set_up_camera(self, enable_preview):
        """
        Configures the camera, preview window and encoder.

        :param enable_preview: enables preview window
        """
        self.__lsize = (self.__lores_width, self.__lores_height)
        self.__picam2 = Picamera2()
        video_config = self.__picam2.create_video_configuration(
            main={"size": (self.__width, self.__height), "format": "RGB888"},
            lores={"size": self.__lsize, "format": "YUV420"})
        self.__picam2.configure(video_config)

        if enable_preview:
            self.__picam2.start_preview(Preview.QT, x=self.__preview_x, y=self.__preview_y,
                                        width=self.__preview_width, height=self.__preview_height)

        self.__encoder = H264Encoder(1000000, repeat=True)
        self.__encoder.output = CircularOutput()
        self.__picam2.encoder = self.__encoder

    def __set_zoom_factor(self):
        """
        Sets the zoom factor of the camera.
        """
        size = self.__picam2.capture_metadata()['ScalerCrop'][2:]
        self.__picam2.capture_metadata()
        size = [int(s * self.__zoom_factor) for s in size]
        offset = [(r - s) // 2 for r, s in zip(self.__picam2.sensor_resolution, size)]
        self.__picam2.set_controls({"ScalerCrop": offset + size})

    def __delete_recording(self, file_path):
        """
        Deletes video, if the appropriate command line argument is supplied.
        :param file_path: file to delete
        """
        if self.__delete_local_recordings:
            self.log_info(f"Deleting local file: {file_path}")
            os.remove(file_path)

    def __send_email(self, file_path):
        """
        Sends an email with the recording attached.
        :param file_path: Path of the recording to send
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.__email_username
            msg['To'] = self.__recipient
            msg['Subject'] = f"Motion detected at {datetime.datetime.now()}"

            with open(file_path, 'rb') as attachment_file:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(attachment_file.read())
                encoders.encode_base64(attachment)
                attachment.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
                msg.attach(attachment)

            with smtplib.SMTP_SSL(self.__smtp_server, self.__smtp_port, timeout=10) as server:
                server.login(self.__email_username, self.__email_password)
                server.sendmail(self.__email_username, self.__recipient, msg.as_string())
                self.log_info(f"Sent email with attachment {file_path}")
        except (smtplib.SMTPException, socket.timeout) as e:
            self.log_error(f"Failed to send email with attachment {file_path}: {e}")

    def __upload_file(self, file_path):
        """
        Sends the recording via email and deletes it.
        :param file_path:
        """
        if self.__email_username and self.__recipient and self.__email_password:
            self.__send_email(file_path)
        self.__delete_recording(file_path)

    def stop(self):
        """
        Stops the encoder and exits the application.
        """
        self.__picam2.stop_encoder()
        sys.exit(1)

    def log_debug(self, message):
        logging.debug(f"{Fore.LIGHTBLUE_EX} {message} {Style.RESET_ALL}")

    def log_info(self, message):
        logging.info(f"{Fore.LIGHTYELLOW_EX} {message} {Style.RESET_ALL}")

    def log_warning(self, message):
        logging.warning(f"{Fore.YELLOW} {message} {Style.RESET_ALL}")

    def log_error(self, message):
        logging.error(f"{Fore.RED} {message} {Style.RESET_ALL}")

    def log_debug_as_info(self, message):
        logging.info(f"{Fore.LIGHTBLUE_EX} {message} {Style.RESET_ALL}")

    def log_movement_start(self, message):
        logging.info(f"{Back.RED} {message} {Style.RESET_ALL}")
    def log_movement_end(self, message):
        logging.info(f"{Back.GREEN} {message} {Style.RESET_ALL}")

    def log_at_interval(self, message):
        if self.__tick == self.__display_interval:
                self.log_debug_as_info(message)
                self.__tick = 0
        else:
            self.__tick += 1

    def display_diff_stats(self):
        diff_sum = 0
        self.__diff_min = 9999
        self.__diff_max = 0
        for value in self.__diff_history:
            if value < self.__diff_min:
                self.__diff_min = value
            if value > self.__diff_max:
                self.__diff_max = value
            diff_sum += value

        iterations = len(self.__diff_history)
        self.__diff_average = diff_sum / len(self.__diff_history)
        self.log_at_interval(f"Diff Stats ({iterations} iterations): AVG: {self.__diff_average} | MIN: {self.__diff_min} | MAX: {self.__diff_max}")

    def store_diff_history(self, diff):
        if len(self.__diff_history) == self.__diff_history_count:
            new_diff_history = [diff]
            pos = 0
            for value in self.__diff_history:
                if 1 <= pos <= self.__diff_history_count - 1:
                    new_diff_history.append(value)
                pos += 1
            self.__diff_history.clear()
            for value in new_diff_history:
                self.__diff_history.append(value)
            new_diff_history.clear()
        else:
            self.__diff_history.append(diff)
        self.display_diff_stats()


if __name__ == "__main__":
    command_line_arguments = parse_command_line_arguments()
    motion_detector = MotionDetector(command_line_arguments)
    signal.signal(signal.SIGINT, command_line_handler)
    motion_detector.start()
