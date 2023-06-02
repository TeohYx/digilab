import speech_recognition as sr
import pyaudio

# Create a recognizer object
r = sr.Recognizer()

# Use the microphone as the audio source
with sr.Microphone() as source:
    print("Speak something...")
    audio = r.listen(source)

try:
    # Use the specified speech recognition engine to convert speech to text
    text = r.recognize_google(audio)  # Replace with your desired engine

    # Print the recognized text
    print("Recognized text: " + text)

except sr.UnknownValueError:
    print("Could not understand audio.")
except sr.RequestError as e:
    print("Error: {0}".format(e))