import logging
import asyncio
from utils import commands, TICK_FREQ_COMMAND, START_COMMAND

class EyeTracker:
    def __init__(self, host, port, user_data="") -> None:
        self.host = host
        self.port = port
        self.user_data = user_data

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        logging.info(f"Connected to {self.host}:{self.port}")

    async def start_stream(self):
        await self.connect()
        for command in commands:
            self.writer.write(command.encode())
        await self.set_user_data()
        await self.send_command(START_COMMAND)
        return self.reader

    async def get_msg(self):
        if not self.reader:
            logging.error("Not connected")
        try:
            data = await asyncio.wait_for(self.reader.readline(), 1)
        except TimeoutError:
            return None
        data = data.decode()
        try:
            msg = ET.fromstring(data)
            return msg
        except Exception:
            return None

    async def stop_stream(self):
        logging.info("Close connection")
        self.writer.close()
        await self.writer.wait_closed()

    async def get_tick_frequency(self):
        await self.connect()
        await self.send_command('<GET ID="TIME_TICK_FREQUENCY" />\r\n')
        msg = await self.get_msg()
        if msg is None:
            raise Exception("Could not get tick frequency")
        if msg.tag == "ACK" and "FREQ" in msg.attrib:
            freq = msg.get("FREQ")
            if freq is not None:
                return float(freq)
            else:
                raise Exception("Could not get tick frequency")

    async def start_calibration(self):
        await self.connect()
        show_calibration_command = '<SET ID="CALIBRATE_SHOW" STATE="1" />\r\n'.encode()
        start_calibration_command = (
            '<SET ID="CALIBRATE_START" STATE="1" />\r\n'.encode()
        )
        self.writer.write(show_calibration_command)
        self.writer.write(start_calibration_command)
        await self.writer.drain()
        return self.reader

    async def set_user_data(self):
        command = f'<SET ID="USER_DATA" VALUE="{self.user_data}" />\r\n'
        await self.send_command(command)

    async def send_command(self, command: str):
        self.writer.write(command.encode())
        await self.writer.drain()