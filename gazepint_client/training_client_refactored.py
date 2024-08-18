from os import name
import xml.etree.ElementTree as ET
import keyboard
from eyetracker import EyeTracker
from eyetrackermessagehandler import EyeTrackerMessageHandler
from config import HOST, PORT
import asyncio
import cv2
import argparse

from datahandler import DataHandler

import argparse
from synchelper import synchelper_singleton
from trainer_types import Subject
import sys
from calibration import Calibration



class EyeStrams:
    def __init__(self, label, data_handler) -> None:
        self.running = True
        self.calibration_task = None
        self.label = label
        self.data_handler = data_handler

    def start_calibration(self):
        logging.info("Starting stream")
        self.calibration_task = asyncio.create_task(self.run_stream())
        self.running = True

    async def run_stream(self):
        eyetracker = EyeTracker()
        await eyetracker.start_stream()
        self.running = True
        try:
            while not keyboard.is_pressed('q'):
                await asyncio.sleep(0)
                msg = await eyetracker.get_msg()
                if msg is not None:
                    if msg.tag == "REC":
                        attributes = msg.attrib
                        attributes['LABEL'] = self.label
                        self.data_handler.write_stream_data(msg.attrib)
            print("Data collected and saved")
        except Exception as e:
            logging.debug("Calibration Exception ", e)
        await eyetracker.stop_stream()
        self.running = False
        logging.info("Calibration Finished ")

    def is_finished(self):
        if self.calibration_task is not None: 
            if self.calibration_task.done():
                self.calibration_task.cancel()
                self.running = False
                self.calibration_task = None
                return True
        return self.running is False


async def main(label:str, subject:str):
    calibration = Calibration()
    cal = await calibration.run_calibration_task()
    data_handler: DataHandler = DataHandler(Subject(name=subject, images=[]))
    
    if calibration.is_finished():
        if cv2.EVENT_LBUTTONDBLCLK: #https://stackoverflow.com/questions/39235454/how-to-know-if-the-left-mouse-click-is-pressed
            print("starting data collection")
            et = EyeTracker(HOST, PORT, user_data='test_subject')
            et_tick_freq = await et.get_tick_frequency()
            synchelper_singleton.set_et_tick_freq(et_tick_freq)
            es = EyeStrams(label=label, data_handler=data_handler)
            result = await es.run_stream()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--subject', type=str, required=True)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(label=args.label, subject=args.subject)))
