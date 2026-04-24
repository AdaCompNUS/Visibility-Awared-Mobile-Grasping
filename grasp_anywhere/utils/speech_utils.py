#!/usr/bin/env python3
import os

from openai import OpenAI
from pydub import AudioSegment

from grasp_anywhere.utils.logger import log


class SpeechHandler:
    """
    Utility class for handling text-to-speech and speech-to-text functionality.
    Can be shared across different planners that need speech capabilities.
    """

    def __init__(self, logger=None):
        """
        Initialize the speech handler.

        Args:
            logger (callable, optional): Logger function to use. Defaults to print.
        """
        self.client = OpenAI()
        log.info("SpeechHandler initialized.")

    def text_to_voice(self, text):
        """
        Convert text to speech using OpenAI TTS.

        Args:
            text (str): The text to convert to speech.

        Returns:
            AudioSegment: The generated audio.
        """
        output_file = "output.mp3"
        with self.client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            input=text,
        ) as response:
            response.stream_to_file(output_file)

        voice_audio = AudioSegment.from_mp3(output_file)
        os.remove(output_file)
        return voice_audio

    def speech_to_text(self):
        """
        Convert speech to text. Currently returns a simulated response.

        Returns:
            str: The transcribed text in lowercase.
        """
        log.info("Please type your answer: ")
        # answer = input() # TODO: Uncomment this when we have a way to get the answer from the user
        answer = "yes"  # Default answer for simulation
        return answer.lower()

    def say(self, text):
        """
        Say the given text using text-to-speech.

        Args:
            text (str): The text to speak.

        Returns:
            bool: True if successful, False otherwise.
        """
        # try:
        #     log.info("Converting text to speech...")
        #     question_audio = self.text_to_voice(text)
        #     log.info("Playing speech...")
        #     play(question_audio)
        #     return True
        # except Exception as e:
        #     log.info(f"Error in text-to-speech: {e}")
        #     return False
        print(f"Saying: {text}")
        return True

    def listen(self):
        """
        Listen for speech and convert to text.

        Returns:
            str: The transcribed text, or None if failed.
        """
        try:
            answer_text = self.speech_to_text()
            log.info(f"Transcribed answer: {answer_text}")
            return answer_text
        except Exception as e:
            log.info(f"Error in speech-to-text: {e}")
            return None
