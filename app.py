import re
from flask import Flask, render_template, send_file, jsonify
from flask_socketio import SocketIO, emit
import assemblyai as aai
import threading
import asyncio
import pandas as pd
from io import BytesIO
import os

app = Flask(__name__)
# Initialize SocketIO with eventlet for WebSocket support
socketio = SocketIO(app, async_mode="eventlet")

aai.settings.api_key = '08d0a4c4a98341adbe2f3c0eb224dc69'
transcriber = None  
session_id = None  
transcriber_lock = threading.Lock()

# Data storage for highlighted entities
highlighted_data = []

prompt = """You are a medical transcript analyzer. Your task is to detect and format words/phrases that fit into the following five categories:

Protected Health Information (PHI): Change the font color to red using <span style="color: red;">. This includes personal identifying information such as names, ages, nationalities, gender identities, and organizations.
Medical Condition/History: Highlight the text in light green using <span style="background-color: lightgreen;">. This encompasses any references to illnesses, diseases, symptoms, or conditions.
Anatomy: Italicise the text using <em>. This covers any mentions of body parts or anatomical locations.
Medication: Highlight the text in yellow using <span style="background-color: yellow;">. This includes any references to prescribed drugs, over-the-counter medications, vitamins, or supplements.
Tests, Treatments, & Procedures: Change the font color to green using <span style="color: darkblue;">. This involves any mentions of medical tests, treatments, or procedures performed or recommended.
You will receive a medical transcript along with a list of entities detected by AssemblyAI. Use the detected entities to format the text accordingly and also identify and format any additional relevant medical conditions, anatomical references, medications, and procedures that are not included in the detected entities. Do not return anything except the original transcript which has been formatted. Don't write any additional prefacing text like "here's the formatted transcript" """

def on_open(session_opened: aai.RealtimeSessionOpened):
    global session_id
    session_id = session_opened.session_id
    print("Session ID:", session_id)

def on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        socketio.emit('transcript', {'text': transcript.text})
        asyncio.run(analyze_transcript(transcript.text))
    else:
        # Emit the partial transcript to be displayed in real-time
        socketio.emit('partial_transcript', {'text': transcript.text})

async def analyze_transcript(transcript):
    global highlighted_data

    # Call to LLM with transcript text and retrieve formatted response
    result = aai.Lemur().task(
        prompt, 
        input_text=transcript,
        final_model=aai.LemurModel.claude3_5_sonnet
    )
    
    # Emit the formatted text to the client
    socketio.emit('formatted_transcript', {'text': result.response})

    # Extract and store highlighted data
    # Regular expressions to extract categorized text from the formatted response
    categories = {
        "PHI": re.findall(r'<span style="color: red;">(.*?)<\/span>', result.response),
        "Medical Condition/History": re.findall(r'<span style="background-color: lightgreen;">(.*?)<\/span>', result.response),
        "Anatomy": re.findall(r'<em>(.*?)<\/em>', result.response),
        "Medication": re.findall(r'<span style="background-color: yellow;">(.*?)<\/span>', result.response),
        "Tests, Treatments, & Procedures": re.findall(r'<span style="color: darkblue;">(.*?)<\/span>', result.response)
    }

    # Append extracted data to highlighted_data
    highlighted_data.append({
        "PHI": ', '.join(categories["PHI"]),
        "Medical Condition/History": ', '.join(categories["Medical Condition/History"]),
        "Anatomy": ', '.join(categories["Anatomy"]),
        "Medication": ', '.join(categories["Medication"]),
        "Tests, Treatments, & Procedures": ', '.join(categories["Tests, Treatments, & Procedures"])
    })

def on_error(error: aai.RealtimeError):
    print("An error occurred:", error)

def on_close():
    global session_id
    session_id = None
    print("Closing Session")

def transcribe_real_time():
    global transcriber  
    transcriber = aai.RealtimeTranscriber(
        sample_rate=16_000,
        on_data=on_data,
        on_error=on_error,
        on_open=on_open,
        on_close=on_close
    )

    transcriber.connect()

    microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
    transcriber.stream(microphone_stream)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download_excel')
def download_excel():
    global highlighted_data

    # Convert highlighted data to a DataFrame
    df = pd.DataFrame(highlighted_data)
    
    # Save DataFrame to an in-memory Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Highlighted Data")
    
    output.seek(0)

    # Send the file as a download
    return send_file(output, download_name="trancribed_data.xlsx", as_attachment=True)

@socketio.on('toggle_transcription')
def handle_toggle_transcription():
    global transcriber, session_id  
    with transcriber_lock:
        if session_id:
            if transcriber:
                print("Closing transcriber session")
                transcriber.close()
                transcriber = None
                session_id = None  
        else:
            print("Starting transcriber session")
            threading.Thread(target=transcribe_real_time).start()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Use PORT from environment or default to 5000
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
