import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
from keras.models import load_model
from pygame import mixer
import librosa
import librosa.display

model=load_model("./best_model.h5")
w=tk.Tk()
w.geometry("800x600")
w.title("SPEECH EMOTION RECOGNIZATION")
w.configure(background='#ABCDAB')
label=Label(w,background="#AABBCC",font=('arial',15,'bold'))
def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    n_steps = pitch_factor
    return librosa.effects.pitch_shift(data, n_steps=n_steps, sr=sampling_rate)
def play_audio(audio_file):
        if not mixer.get_init():
            mixer.init()

        # Play the audio
        mixer.music.load(audio_file)
        mixer.music.play()
def Detect(file_path):
    global Label_packed
    global sample_rate
    data, sample_rate = librosa.load("C:/Users/pulag/Downloads/03-01-01-01-01-01-01.wav", duration=2.5, offset=0.6)
    res1 = extract_features(data)
    result = np.array(res1)
        
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically
    arr=['surprise', 'neutral', 'disgust', 'fear', 'sad', 'calm', 'happy', 'angry']
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3)) # stacking vertically
    op=model.predict(result)[0].argmax()
    label.configure(foreground="#543210",text=arr[op])

    

def show_detect(file_path):
    detect_b=Button(w,text="detect audio",command=lambda:Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#120314",foreground="white",font=("arial",10,"bold"))
    detect_b.place(relx=0.79,rely=0.46)
    

def upload_aud():
    try:
        file_path=filedialog.askopenfilename()
        play_button = tk.Button(w, text="Play Audio", command=lambda:play_audio(file_path))
        play_button.pack(pady=20)
        show_detect(file_path)
    except:
        pass
upload=Button(w,text="Upload Audio",command=upload_aud,padx=10,pady=5)
upload.configure(background="#453212",foreground="white",font=("arial",10,"bold"))
upload.pack(side="bottom")
label.pack(side="bottom",expand=True)


heading=Label(w,text="SPEECH EMOTION RECOGNIZATION",pady=20,font=("arial",20,"bold"))
heading.configure(background="#BBCCAA",foreground="#234312")
heading.pack()
w.mainloop()